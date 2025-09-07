# Requirements Document

## Introduction

This specification addresses critical improvements to the dxee-china application's research functionality by streamlining the frontend interface and enhancing the Deep Think feature's reliability and performance. The feature focuses on removing redundant UI elements while implementing robust MongoDB-based HTML caching and session-resilient processing for the Deep Think research mode.

The enhancement aims to simplify the user experience by consolidating similar functionality while ensuring uninterrupted research processing even when frontend sessions disconnect, maintaining the high-quality research output users expect from the platform.

## Project Description (User Input)
2点、コードを直したいです。1) html frontend の"Google Deep"ボタンと"DeepSeek"ボタンは消します。
html frontendの"Search" "Deep Search" "Deep Think"ボタンは残し、これらのボタンが押されて実行されるコードは残し、
"Google Deep"ボタンと"DeepSeek"ボタンからだけ実行されるコードは消してください。
2) 次に、"Deep Think"ボタンを押されたコードは、途中でStreaming が途切れる問題があり、またコードフローも
複雑すぎます。"Deep Think"ボタンを押されたコードは @/backend/test_deepseek_advanced_web_research4_01.py を実行しするフローとほぼ同じで良く、research_results_20250904_104734.json の結果と同等の結果を持つものを期待しています。また、
Serper からのhtml retrieve の前に、以前にserper から同じURLに対して　html retrive したものがあるかMongoDBをCheckして、
期間内に過去にhtml retrieve した情報がMongoDBにあれば、html retrive をせずに　MongoDBのものをDeepSeek APIに渡すアルゴリズムにしてください。 また、html retrieve したら、MongoDBに格納してください。
"Deep Think"ボタンを押されたコードは、Streaming 出力をfrontend html に対して行いますが、もしfrontend html のsessionが切れていても、Deep Thinking mode は続行し、User のChat History にAnswer とConclusionなどを書き込むところまで処理が継続するようにしてください。

## Requirements

### Requirement 1: Frontend Button Interface Cleanup
**User Story:** As a user, I want a simplified research interface with only essential buttons, so that I can easily access the core research functionality without confusion from redundant options.

#### Acceptance Criteria
1. WHEN the chat interface loads THEN the system SHALL display only "Search", "Deep Search", and "Deep Think" buttons
2. WHEN the chat interface loads THEN the system SHALL NOT display "Google Deep" or "DeepSeek" buttons
3. WHEN a user clicks "Search" button THEN the system SHALL execute the existing search functionality without modification
4. WHEN a user clicks "Deep Search" button THEN the system SHALL execute the existing deep search functionality without modification
5. WHEN a user clicks "Deep Think" button THEN the system SHALL execute the enhanced deep think functionality
6. WHERE the codebase contains functions exclusive to "Google Deep" or "DeepSeek" buttons THE system SHALL remove those code sections
7. WHERE the codebase contains shared functions used by multiple buttons THE system SHALL preserve those functions for remaining button functionality

### Requirement 2: HTML Content Caching System
**User Story:** As a system administrator, I want HTML content from Serper API calls to be cached in MongoDB with expiration management, so that the system can reduce external API calls and improve response times while maintaining data freshness.

#### Acceptance Criteria
1. WHEN the system needs to retrieve HTML content from a URL THEN it SHALL first check MongoDB for existing cached content
2. IF cached HTML content exists for a URL AND the content is within the configured expiration period THEN the system SHALL use the cached content instead of making a Serper API call
3. IF cached HTML content does not exist for a URL OR the cached content has expired THEN the system SHALL retrieve fresh content via Serper API
4. WHEN HTML content is retrieved via Serper API THEN the system SHALL store the content in MongoDB with timestamp and URL as key
5. WHERE MongoDB contains expired HTML cache entries THE system SHALL include cache cleanup mechanisms during processing
6. WHEN storing HTML content in MongoDB THEN the system SHALL include metadata fields: URL, retrieval_timestamp, content_hash, and expiration_date
7. IF Serper API call fails AND valid cached content exists (even if expired) THEN the system SHALL fall back to using the cached content with appropriate logging

### Requirement 3: Deep Think Flow Simplification
**User Story:** As a user, I want the Deep Think feature to have a reliable and streamlined processing flow based on proven algorithms, so that I can consistently receive high-quality research results without interruptions or complexity.

#### Acceptance Criteria
1. WHEN a user triggers Deep Think mode THEN the system SHALL follow the workflow pattern from test_deepseek_advanced_web_research4_01.py
2. WHEN Deep Think processing begins THEN the system SHALL generate results equivalent in structure and quality to research_results_20250904_104734.json
3. WHEN Deep Think executes web searches THEN the system SHALL use the MongoDB HTML caching system for content retrieval
4. WHEN Deep Think processes multiple URLs THEN the system SHALL handle requests concurrently while respecting API rate limits
5. IF the processing workflow encounters an error THEN the system SHALL implement graceful degradation and continue with available data
6. WHEN Deep Think completes processing THEN the system SHALL generate comprehensive analysis including statistical summaries and source attribution
7. WHERE the current Deep Think implementation has complex branching logic THE system SHALL replace it with the simplified linear workflow

### Requirement 4: Session-Resilient Processing
**User Story:** As a user, I want my Deep Think research to continue processing even if my browser connection is interrupted, so that I don't lose research progress and receive complete results when I reconnect.

#### Acceptance Criteria
1. WHEN Deep Think processing starts THEN the system SHALL create a persistent processing session in MongoDB independent of the frontend connection
2. WHILE Deep Think is processing THE system SHALL continue execution even if the frontend WebSocket/SSE connection is lost
3. WHEN frontend connection is lost during processing THEN the system SHALL continue processing and store all results in the user's chat history
4. WHEN Deep Think processing completes THEN the system SHALL save the complete analysis and conclusion to the user's MongoDB chat history
5. IF the frontend reconnects during processing THEN the system SHALL resume streaming progress updates from the current processing step
6. WHEN processing completes while frontend is disconnected THEN the system SHALL ensure the final results are available when the user returns
7. WHERE processing encounters session timeout THE system SHALL implement configurable timeout handling with appropriate user notification
8. WHEN Deep Think completes successfully THEN the system SHALL mark the processing session as complete with success status in MongoDB

### Requirement 5: Streaming Output Enhancement
**User Story:** As a user, I want to see real-time progress updates during Deep Think processing with reliable streaming, so that I can monitor the research progress and understand what the system is working on.

#### Acceptance Criteria
1. WHEN Deep Think processing begins THEN the system SHALL establish reliable SSE (Server-Sent Events) streaming to the frontend
2. WHILE Deep Think is processing THE system SHALL send progress updates including step description, completion percentage, and intermediate results
3. WHEN each major processing step completes THEN the system SHALL stream a structured progress update with step identifier and status
4. IF streaming connection is interrupted THEN the system SHALL attempt to reconnect and resume streaming from the current processing state
5. WHEN streaming cannot be established or maintained THEN the system SHALL continue processing and ensure results are saved to chat history
6. WHEN processing generates intermediate insights THEN the system SHALL stream those insights while continuing the workflow
7. WHERE streaming fails repeatedly THE system SHALL not implement retry logic.  The user can access the chat history later.
8. WHEN final results are ready THEN the system SHALL stream the complete analysis followed by a completion event

### Requirement 6: Integration with Existing Chat System
**User Story:** As a user, I want Deep Think results to integrate seamlessly with my chat history and conversation context, so that I can reference research results in ongoing conversations and maintain conversation continuity.

#### Acceptance Criteria
1. WHEN Deep Think processing completes THEN the system SHALL save results to the user's current chat conversation in MongoDB
2. WHEN saving to chat history THEN the system SHALL include the original query, processing metadata, and complete analysis results
3. IF Deep Think is triggered within an existing conversation THEN the system SHALL maintain conversation context and threading
4. WHEN results are saved to chat history THEN the system SHALL include source attribution, confidence metrics, and statistical summaries
5. WHERE Deep Think results reference external sources THE system SHALL store clickable URLs and source metadata
6. WHEN a user views their chat history THEN the system SHALL display Deep Think results with proper formatting and structure
7. IF the user shares their conversation THEN the system SHALL include Deep Think results in the shared content with appropriate privacy controls
