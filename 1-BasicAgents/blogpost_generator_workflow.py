from pydantic import BaseModel, Field
from typing import Optional, List, Iterator
from agno.workflow import Workflow, RunResponse, RunEvent
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.utils.log import logger
import os
import json
from dotenv import load_dotenv

load_dotenv()

gemini = Gemini(
    api_key=os.getenv("GEMINI_API_KEY"),
    id="gemini-2.5-pro-exp-03-25"
)

class NewsArticle(BaseModel):
    title: str = Field(description="Title of the article")
    url: str = Field(description="URL of the news article")
    summary: Optional[str] = Field(description="Summary of the news article if available")
    
    
class SearchResults(BaseModel):
    articles: List[NewsArticle]
    
class BlogPostGenerator(Workflow):
    searcher = Agent(
        model=gemini,
        tools=[DuckDuckGoTools()],
        instructions=[
            "Given a topic, search for the top 5 articles.",
            "For each article, provide the following in a clear format:",
            "1. Title of the article",
            "2. URL of the article",
            "3. A brief summary if available",
            "Use the DuckDuckGo search tools to find relevant articles."
        ],
        add_datetime_to_instructions=True,
        structured_outputs=False, 
        debug_mode=True,
        show_tool_calls=True
    )
    
    writer = Agent(
        model=gemini,
        instructions=[
            "You will be provided with a topic and a list of top articles.",
            "Generate a New York Times style blog post with catchy sections based on the articles.",
            "Include key takeaways and always cite the sources."
        ],
        debug_mode=True,
        markdown=True,
    )
    
    def run(self, topic:str, use_cache: bool=True) -> Iterator[RunResponse]:
        """
        Run the workflow to generate a blog post based on the provided topic.
        
        Args:
            topic (str): The topic for the blog post.
            use_cache (bool): Whether to use cached results for the search.
            
        Yields:
            RunResponse: The generated blog post.
        """
        
        logger.info(f"Generating blog post for topic: {topic}")
        logger.info(f"Using cache: {use_cache}")
        
        # Step 1: Check if blog post is already cached
        if use_cache:
            cached_blog_post = self._get_cached_blog_post(topic)
            if cached_blog_post:
                logger.info(f"Using cached blog post for topic: {topic}")
                yield RunResponse(content=cached_blog_post,event=RunEvent.workflow_completed)
                return
            
        # Step 2: Search for articles on the topic
        search_results = self._get_search_results(topic)
        if not search_results or len(search_results.articles) == 0:
            logger.error(f"No search results for topic: {topic}")
            yield RunResponse(content="No search results found for topic: {topic}", event=RunEvent.workflow_completed)
            return
        
        # Step 3: Write the blog post using the search results
        yield from self._write_blog_post(topic, search_results)

        
    def _add_blog_post_to_cache(self, topic:str, blog_post: Optional[str]) -> None:
        logger.info(f"Caching blog post for : {topic}")
        
        self.session_state.setdefault('blog_posts', {})
        self.session_state['blog_posts'][topic] = blog_post
        
        logger.info(f"Blog post cached succesfully for topic: {topic}")
        
    def _get_cached_blog_post(self,topic:str) -> Optional[str]:
        logger.info(f"Checking cached for blog post on topic: {topic}")
        
        cached_post = self.session_state.get('blog_posts', {}).get(topic)
        
        if cached_post:
            logger.info(f"Found cached blog post for topic: {topic}")
        else:
            logger.info(f"No cached blog post found for topic: {topic}")
            
        return cached_post
    
    def _get_search_results(self, topic: str) -> Optional[SearchResults]:
        MAX_ATTEMPTS = 3
        for attempt in range(MAX_ATTEMPTS):
            try:
                logger.info(f"Attempt {attempt + 1}: Searching for articles on topic: {topic}")
                response = self.searcher.run(topic)
                
                if response and response.content:
                    # Parse the unstructured response into articles
                    articles = []
                    # Split the content by newlines and look for numbered items
                    lines = response.content.split('\n')
                    current_article = {}
                    
                    for line in lines:
                        if 'Title:' in line:
                            if current_article:
                                articles.append(NewsArticle(**current_article))
                                current_article = {}
                            current_article['title'] = line.split('Title:')[1].strip()
                        elif 'URL:' in line:
                            current_article['url'] = line.split('URL:')[1].strip()
                        elif 'Summary:' in line:
                            current_article['summary'] = line.split('Summary:')[1].strip()
                    
                    # Add the last article if exists
                    if current_article:
                        articles.append(NewsArticle(**current_article))
                    
                    if articles:
                        logger.info(f"Found {len(articles)} articles on attempt {attempt + 1}")
                        return SearchResults(articles=articles)
                
                logger.warning(f"Attempt {attempt + 1} Invalid or empty response")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed with error: {e}")
                
        logger.error(f"Failed to get search results for {topic} after {MAX_ATTEMPTS} attempts")
        return None
    
    def _write_blog_post(self, topic:str, search_results: SearchResults) -> Iterator[RunResponse]:
        logger.info(f"Writing blog post for topic: {topic}")
        
        writer_input = {
            "topic": topic,
            "articles": [article.model_dump() for article in search_results.articles]
        }
        
        logger.info(f"Input prepared for writer agent: {json.dumps(writer_input, indent=4)}")
        
        yield from self.writer.run(json.dumps(writer_input,indent=4), stream=True)
        
        self._add_blog_post_to_cache(topic, self.writer.run_response.content)
        
        
        
if __name__ == "__main__":
    from rich.prompt import Prompt
    from agno.storage.workflow.sqlite import SqliteWorkflowStorage
    from agno.utils.pprint import pprint_run_response
    
    # Getting the topic from the user
    topic = Prompt.ask("[bold]Enter a topic for the blog post[/bold]\n")
    
    url_safe_topic = topic.lower().replace(" ", "-")
    
    generate_blog_post = BlogPostGenerator(
        session_id=f'generate_blog_post_{url_safe_topic}',
        storage=SqliteWorkflowStorage(
            table_name='generator_blog_post_workflows',
            db_file='/storage/workflows.db',
            
        ),
    )
    
    response = generate_blog_post.run(topic=topic, use_cache=False)
    
    pprint_run_response(response,markdown=False)

