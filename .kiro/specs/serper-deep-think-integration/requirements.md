# Requirements Document

## Introduction

The serper-deep-think-integration feature enhances the existing dxee-china chat application by integrating advanced web research capabilities with deep reasoning logic. This feature replicates the algorithm from test_deepseek_advanced_web_search4_01.py and integrates it into the main application through a new "deep-think" button interface. The integration leverages replicated Serper API logic (direct HTTP calls to Serper endpoints) for web search functionality and incorporates Jan framework's deep-thinking logic to provide comprehensive, reasoned responses to user queries.

This feature adds significant value to the bidirectional information bridge by enabling users to access sophisticated research capabilities directly within the chat interface, combining real-time web search with advanced AI reasoning for enhanced information analysis and synthesis.

## Project Description (User Input)
I would like to replicate the algorithm implemented in test_deepseek_advanced_web_search4_01.py to main program, which I believe it starts from ./backend/app/tornado_main.py.
In the file test_deepseek_advanced_web_search4_01.py, I implemented the web search and deep-thinking logic from Jan under /Users/jun77/Documents/Dropbox/a_root/code/jan/jan/.
and using replicated Serper API logic (inspired by serper-mcp patterns but implemented as direct HTTP calls).
I would like to implement these algorithm to the button on the screen where a user input questions and click 'deep-think' button, the algorithm will start using the web search and deep-thinking logic from Jan through direct calls to Serper API endpoints.
The goal scenario is , when a user input questions and click 'deep-think' button,
the system will think of queries and call to serper api and retrieve search results
from serper api and think of answers to the quetion thru deep-thinking/deep-reasoning
and reply the answer and summarized answer to the chat on the web screen.

## Requirements

### Requirement 1: Deep-Think Button User Interface
**User Story:** As a chat user, I want a "deep-think" button available in the chat interface, so that I can trigger advanced research mode for complex queries requiring web search and deep reasoning.

#### Acceptance Criteria
1. WHEN a user views the chat interface THEN the system SHALL display a "deep-think" button alongside the standard send message controls
2. WHEN a user enters a message in the chat input field THEN the system SHALL enable the "deep-think" button for user interaction
3. WHEN a user clicks the "deep-think" button THEN the system SHALL initiate the deep research workflow instead of standard chat processing
4. IF the user input field is empty WHEN the user clicks "deep-think" button THEN the system SHALL display a validation message requesting user input
5. WHILE the deep-think process is running THE system SHALL disable the "deep-think" button to prevent duplicate requests

### Requirement 2: Query Generation and Processing
**User Story:** As a user initiating deep-think mode, I want the system to intelligently generate relevant search queries from my input, so that comprehensive web research can be conducted on my behalf.

#### Acceptance Criteria
1. WHEN a user activates deep-think mode THEN the system SHALL analyze the user input to generate multiple relevant search queries
2. WHEN generating search queries THEN the system SHALL create 3-5 distinct queries covering different aspects of the user's question
3. IF the user input contains multiple topics THEN the system SHALL generate queries addressing each distinct topic area
4. WHEN query generation is complete THEN the system SHALL log the generated queries for debugging and monitoring purposes
5. WHILE generating queries THE system SHALL display a progress indicator showing "Analyzing question and generating search queries"

### Requirement 3: Direct Serper API Integration
**User Story:** As a system processing deep-think requests, I want to retrieve comprehensive web search results via direct Serper API calls, so that current and relevant information can be gathered for analysis.

#### Acceptance Criteria
1. WHEN search queries are generated THEN the system SHALL execute each query against Serper API endpoints using direct HTTP requests
2. WHEN calling the Serper API THEN the system SHALL handle authentication via API key headers and request formatting according to the Serper API specification
3. IF the Serper API returns an error THEN the system SHALL log the error and attempt retry with exponential backoff (maximum 3 attempts)
4. WHEN API calls are successful THEN the system SHALL collect and aggregate search results from all queries
5. WHILE executing API calls THE system SHALL display progress updates showing "Searching web for relevant information"
6. WHEN API response exceeds rate limits THEN the system SHALL queue remaining requests and notify the user of potential delays

### Requirement 4: Jan Deep-Thinking Logic Integration
**User Story:** As a user expecting sophisticated analysis, I want the system to apply Jan framework's deep-thinking algorithms to the collected search results, so that I receive reasoned and synthesized answers rather than raw search data.

#### Acceptance Criteria
1. WHEN web search results are collected THEN the system SHALL apply Jan framework's deep-thinking algorithms to analyze the content
2. WHEN processing with Jan logic THEN the system SHALL evaluate information relevance, credibility, and relationship between sources
3. WHEN deep-thinking analysis is complete THEN the system SHALL generate reasoning chains explaining how conclusions were reached
4. IF conflicting information is found in search results THEN the system SHALL identify contradictions and provide balanced analysis
5. WHILE deep-thinking processing is active THE system SHALL stream progress updates showing "Analyzing and reasoning through collected information"
6. WHEN reasoning is complete THEN the system SHALL prepare both detailed analysis and concise summary versions

### Requirement 5: Real-Time Progress Streaming
**User Story:** As a user waiting for deep-think results, I want to see real-time progress updates during the research process, so that I understand the system is actively working and can estimate completion time.

#### Acceptance Criteria
1. WHEN deep-think mode is activated THEN the system SHALL establish a Server-Sent Events (SSE) stream for progress updates
2. WHEN each major processing step begins THEN the system SHALL send progress updates with step description and completion percentage
3. WHILE processing is ongoing THE system SHALL send progress updates at least every 10 seconds to maintain user engagement
4. WHEN processing steps complete THEN the system SHALL update progress indicators with completed status and next step information
5. IF processing encounters errors THEN the system SHALL stream error notifications with recovery status
6. WHEN the entire workflow completes THEN the system SHALL send a final completion event with full results

### Requirement 6: Answer Generation and Summarization
**User Story:** As a user receiving deep-think results, I want comprehensive answers along with concise summaries, so that I can quickly understand key points while having access to detailed analysis when needed.

#### Acceptance Criteria
1. WHEN deep-thinking analysis completes THEN the system SHALL generate a comprehensive answer incorporating all relevant findings
2. WHEN generating answers THEN the system SHALL include source citations and confidence indicators for key claims
3. WHEN comprehensive answer is ready THEN the system SHALL create a concise summary highlighting the most important points
4. IF the analysis reveals uncertainty or conflicting information THEN the system SHALL clearly indicate areas of uncertainty in both answer formats
5. WHEN both answer formats are prepared THEN the system SHALL format them using markdown for proper display in the chat interface
6. WHERE multiple topics were addressed THE system SHALL organize answers into logical sections with clear headings

### Requirement 7: Chat Integration and Display
**User Story:** As a user viewing deep-think results, I want the analysis and summary to appear seamlessly in my chat conversation, so that the information is preserved as part of my chat history.

#### Acceptance Criteria
1. WHEN deep-think processing completes THEN the system SHALL display results as a new message in the chat conversation
2. WHEN displaying results THEN the system SHALL clearly distinguish deep-think responses from standard chat responses with visual indicators
3. WHEN results are displayed THEN the system SHALL include both the comprehensive answer and summary in an organized, expandable format
4. IF the response is lengthy THEN the system SHALL provide summary view by default with option to expand to full details
5. WHEN results are saved THEN the system SHALL store both the user's original question and the complete deep-think response in the chat history database
6. WHERE source citations are included THE system SHALL format them as clickable links when applicable

### Requirement 8: Error Handling and Recovery
**User Story:** As a user experiencing system issues during deep-think processing, I want clear error messages and graceful degradation, so that I understand what went wrong and can take appropriate action.

#### Acceptance Criteria
1. IF Serper API is unavailable WHEN deep-think is activated THEN the system SHALL notify the user and suggest trying again later
2. IF Jan framework processing fails THEN the system SHALL fall back to standard search result presentation with error notification
3. WHEN network timeouts occur THEN the system SHALL attempt automatic retry and inform the user of retry attempts
4. IF the entire deep-think process fails THEN the system SHALL preserve the user's original question and suggest alternative approaches
5. WHILE errors are being handled THE system SHALL maintain the chat interface responsiveness for other user actions
6. WHEN recovery is not possible THEN the system SHALL provide clear instructions for reporting the issue and accessing alternative features

### Requirement 9: Performance and Resource Management
**User Story:** As a system administrator, I want the deep-think feature to operate within reasonable resource limits, so that it doesn't impact overall application performance or user experience.

#### Acceptance Criteria
1. WHEN deep-think processes are initiated THEN the system SHALL enforce a maximum processing time of 10 minutes per request
2. WHEN multiple users trigger deep-think simultaneously THEN the system SHALL queue requests to prevent resource exhaustion
3. IF processing time exceeds 8 minutes THEN the system SHALL send a warning notification about potential timeout
4. WHEN system resources are constrained THEN the system SHALL prioritize active user sessions over queued deep-think requests
5. WHILE deep-think is processing THE system SHALL maintain normal chat functionality for other conversations and users
6. WHEN deep-think completes THEN the system SHALL clean up temporary resources and update performance metrics