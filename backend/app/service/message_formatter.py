import re
import html
from typing import Dict, Any

class MessageFormatter:
    """
    Service for formatting chat messages with Markdown and rich text support
    """
    
    def __init__(self):
        # Markdown patterns for basic formatting
        self.markdown_patterns = [
            # Bold text: **text** or __text__
            (r'\*\*(.*?)\*\*', r'<strong>\1</strong>'),
            (r'__(.*?)__', r'<strong>\1</strong>'),
            
            # Italic text: *text* or _text_
            (r'\*(.*?)\*', r'<em>\1</em>'),
            (r'_(.*?)_', r'<em>\1</em>'),
            
            # Code inline: `code`
            (r'`([^`]+)`', r'<code>\1</code>'),
            
            # Links: [text](url)
            (r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>'),
            
            # Strikethrough: ~~text~~
            (r'~~(.*?)~~', r'<del>\1</del>'),
        ]
    
    def format_message(self, message: str, format_type: str = "markdown") -> Dict[str, Any]:
        """
        Format a message with the specified formatting type
        
        Args:
            message (str): The raw message text
            format_type (str): The formatting type ('markdown', 'html', 'plain')
            
        Returns:
            dict: Formatted message with metadata
        """
        if format_type == "markdown":
            return self._format_markdown(message)
        elif format_type == "html":
            return self._format_html(message)
        else:
            return self._format_plain(message)
    
    def _format_markdown(self, message: str) -> Dict[str, Any]:
        """
        Format message as Markdown with HTML output
        
        Args:
            message (str): Raw message text
            
        Returns:
            dict: Formatted message data
        """
        # Escape HTML first to prevent XSS
        escaped_message = html.escape(message)
        
        # Process different markdown elements
        formatted_html = self._process_markdown_blocks(escaped_message)
        formatted_html = self._process_markdown_inline(formatted_html)
        
        return {
            "raw_content": message,
            "formatted_content": formatted_html,
            "format_type": "markdown",
            "has_formatting": formatted_html != escaped_message
        }
    
    def _format_html(self, message: str) -> Dict[str, Any]:
        """
        Format message as HTML (sanitized)
        
        Args:
            message (str): Raw message text
            
        Returns:
            dict: Formatted message data
        """
        # Basic HTML sanitization - allow only safe tags
        safe_tags = ['p', 'br', 'strong', 'b', 'em', 'i', 'u', 'code', 'pre', 
                    'ul', 'ol', 'li', 'blockquote', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        
        # For now, just escape HTML to prevent XSS
        # In production, you'd want to use a proper HTML sanitizer like bleach
        sanitized_html = html.escape(message)
        
        return {
            "raw_content": message,
            "formatted_content": sanitized_html,
            "format_type": "html",
            "has_formatting": False
        }
    
    def _format_plain(self, message: str) -> Dict[str, Any]:
        """
        Format message as plain text
        
        Args:
            message (str): Raw message text
            
        Returns:
            dict: Formatted message data
        """
        # Convert line breaks to HTML breaks for display
        formatted_text = html.escape(message).replace('\n', '<br>')
        
        return {
            "raw_content": message,
            "formatted_content": formatted_text,
            "format_type": "plain",
            "has_formatting": '\n' in message
        }
    
    def _process_markdown_blocks(self, text: str) -> str:
        """
        Process block-level markdown elements
        
        Args:
            text (str): Text to process
            
        Returns:
            str: Text with block elements processed
        """
        lines = text.split('\n')
        processed_lines = []
        in_code_block = False
        in_list = False
        list_type = None
        
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            
            # Handle code blocks
            if line.startswith('```'):
                if not in_code_block:
                    in_code_block = True
                    language = line[3:].strip()
                    processed_lines.append(f'<pre><code class="language-{language}">')
                else:
                    in_code_block = False
                    processed_lines.append('</code></pre>')
                i += 1
                continue
            
            if in_code_block:
                processed_lines.append(line)
                i += 1
                continue
            
            # Handle headers
            if line.startswith('#'):
                level = 0
                for char in line:
                    if char == '#':
                        level += 1
                    else:
                        break
                if level <= 6 and line[level:].strip():
                    header_text = line[level:].strip()
                    processed_lines.append(f'<h{level}>{header_text}</h{level}>')
                    i += 1
                    continue
            
            # Handle blockquotes
            if line.startswith('>'):
                quote_text = line[1:].strip()
                processed_lines.append(f'<blockquote>{quote_text}</blockquote>')
                i += 1
                continue
            
            # Handle lists
            list_match = re.match(r'^(\s*)([-*+]|\d+\.)\s+(.+)$', line)
            if list_match:
                indent, marker, content = list_match.groups()
                is_ordered = marker.endswith('.')
                
                if not in_list:
                    in_list = True
                    list_type = 'ol' if is_ordered else 'ul'
                    processed_lines.append(f'<{list_type}>')
                elif (is_ordered and list_type == 'ul') or (not is_ordered and list_type == 'ol'):
                    processed_lines.append(f'</{list_type}>')
                    list_type = 'ol' if is_ordered else 'ul'
                    processed_lines.append(f'<{list_type}>')
                
                processed_lines.append(f'<li>{content}</li>')
                i += 1
                continue
            else:
                if in_list:
                    processed_lines.append(f'</{list_type}>')
                    in_list = False
                    list_type = None
            
            # Handle horizontal rules
            if re.match(r'^[-*_]{3,}$', line.strip()):
                processed_lines.append('<hr>')
                i += 1
                continue
            
            # Handle paragraphs
            if line.strip():
                processed_lines.append(f'<p>{line}</p>')
            else:
                processed_lines.append('<br>')
            
            i += 1
        
        # Close any open lists
        if in_list:
            processed_lines.append(f'</{list_type}>')
        
        return '\n'.join(processed_lines)
    
    def _process_markdown_inline(self, text: str) -> str:
        """
        Process inline markdown elements
        
        Args:
            text (str): Text to process
            
        Returns:
            str: Text with inline elements processed
        """
        result = text
        
        # Apply inline markdown patterns
        for pattern, replacement in self.markdown_patterns:
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def detect_format_type(self, message: str) -> str:
        """
        Auto-detect the likely format type of a message
        
        Args:
            message (str): Message to analyze
            
        Returns:
            str: Detected format type
        """
        # Check for common markdown patterns
        markdown_indicators = [
            r'\*\*.*?\*\*',  # Bold
            r'__.*?__',      # Bold
            r'\*.*?\*',      # Italic
            r'_.*?_',        # Italic
            r'`.*?`',        # Code
            r'#{1,6}\s',     # Headers
            r'^\s*[-*+]\s',  # Lists
            r'^\s*\d+\.\s',  # Numbered lists
            r'>\s',          # Blockquotes
            r'\[.*?\]\(.*?\)',  # Links
        ]
        
        for pattern in markdown_indicators:
            if re.search(pattern, message, re.MULTILINE):
                return "markdown"
        
        # Check for HTML tags
        if re.search(r'<[^>]+>', message):
            return "html"
        
        return "plain"
