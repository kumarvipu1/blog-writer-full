import asyncio
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.messages import ModelMessage
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_experimental.utilities import PythonREPL
from tabulate import tabulate
from typing_extensions import TypedDict
from typing import Literal, Annotated, List, Optional, Union
# from rich.prompt import Prompt  # Removed - not compatible with web applications
from datetime import datetime
import json
from dataclasses import dataclass, field
from pydantic_ai.models.openai import OpenAIModel
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
import logfire
import re
from tavily import TavilyClient
from contextlib import redirect_stdout
from pydantic_ai.providers.openai import OpenAIProvider
from io import StringIO
from pydantic_graph import Graph, BaseNode, GraphRunContext, End
import requests
import os
from pydantic_graph import GraphRunContext, mermaid
from dotenv import load_dotenv
from datetime import time
import time # Import time for potential delays


logfire.configure(token=os.getenv('LOGFIRE_TOKEN'), scrubbing=False)
model = OpenAIModel('gpt-4.1', provider=OpenAIProvider(api_key=os.getenv('OPENAI_API_KEY')))

@dataclass
class State:
    user_query: str = field(default_factory=str)
    context: str = field(default_factory=str)
    current_date: str = field(default_factory=str)
    blog_title: str = field(default_factory=str)
    sections: list[str] = field(default_factory=list)
    blog_content: dict[str, str] = field(default_factory=dict)
    research_data: str = field(default_factory=str)
    complete_blog: str = field(default_factory=str)
    instructions: list[str] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)
    url_tool_use_counter: int = field(default=0)
    web_scraper_tool_use_counter: int = field(default=0)
    previous_section_content: str = field(default_factory=str)


# Tools

# to get the source urls for the final blog
def get_source_url(ctx: RunContext[State], query: Annotated[str, "The query to search for"]) -> str:
    """Use this tool to get source urls for the query. Later you can use the web_scraper tool to get the content of the urls."""

    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = client.search(query=query, max_results=4, search_depth="advanced")
    scores = [result['score'] for result in results['results']]
    urls = [result['url'] for result in results['results']]
    images = results['images'][:len(urls)]
    ctx.deps.url_tool_use_counter += 1
    if ctx.deps.url_tool_use_counter <= 4:
        return f"Urls:\n{str(urls)}\n\n Tool usage counter: {ctx.deps.url_tool_use_counter} of 4"
    else:
        ctx.deps.url_tool_use_counter = 0
        return f"Urls:\n{str(urls)}\n\n You have used the tool more than 4 times please move on to the getting the content of the urls."

# to get the content of the urls
def web_scraper(ctx: RunContext[State], urls: Annotated[list, "The urls to scrape for more information and data for writing the blog."],
                length: Annotated[int, "The length of the content to scrape"] = 3000) -> str:
    """Pass one url as a string to get more information and data for writing the blog."""
    
    words_per_url = length // len(urls)  # Distribute words evenly across URLs
    text_data = ""
    
    for url in urls:
        loader = WebBaseLoader(url)
        data = loader.load()
        
        for doc in data:
            # Remove HTML/XML tags first
            content = re.sub(r'<[^>]+>', '', doc.page_content)
            
            # Split into paragraphs
            paragraphs = content.split('\n')
            clean_paragraphs = []
            
            for p in paragraphs:
                # Remove special characters and normalize spaces
                cleaned = re.sub(r'[^\w\s]', '', p)  # Keep only alphanumeric and spaces
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # Normalize to single spaces
                cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned)  # Remove non-English characters
                
                # Only keep paragraphs relevant to the query
                if len(cleaned.split()) > 10 and cleaned:
                    clean_paragraphs.append(cleaned)
            
            filtered_content = ' '.join(clean_paragraphs)  # Join all paragraphs into single text
            final_content = ' '.join(filtered_content.split()[:words_per_url])  # Take exact number of words needed
                
            title = doc.metadata.get("title", "")
            text_data += f'{title}\n{final_content}\n\n'
    
    ctx.deps.web_scraper_tool_use_counter += 1
    if ctx.deps.web_scraper_tool_use_counter <= 4:
        return f"Data from the urls:\n{str(urls)}\n\n{text_data} \n Tool usage counter: {ctx.deps.web_scraper_tool_use_counter} of 4"
    else:
        ctx.deps.web_scraper_tool_use_counter = 0
        return f"{text_data}\n\n You have used the tool more than 4 times please move on to the next section."


# Building intro Agent

class PlannerAgentOutput(BaseModel):
    title: str = Field(description="The title of the blog")
    sections: list[str] = Field(description="The sections planned for the blog")
    instructions: list[str] = Field(description="The instructions for the next agent for each section")


planner_agent = Agent(
    model=model,
    deps_type=State,
    result_type=PlannerAgentOutput,
    instrument=True
)

@planner_agent.system_prompt
async def get_planner_agent_system_prompt(ctx: RunContext[State]):
    prompt = f"""
    You are a helpful assistant who is a blog writer and planner.
    You also set the tone for the blog by providing important keypoints and instructions for the next agent who is also working on the same blog. 

    Your goal is to plan the sections of the blog and provide the instructions for the next agent for each section.

    **Instructions:**
    - Analyse the user query and the context provided by the user.
    - Decide the title of the blog.
    - Plan the sections of the blog such as Introduction, <section_name>, <section_name>, ..., Conclusion etc. in a maximum of 5 to 7 sections depending on the complexity of the topic.
    - Provide the instructions for the next agent for each section like search <section_name> for more information and write introduction for the section in tone ...., write the content of the section in tone ...., write the conclusion for the section in tone ...., etc.

    **Input Data:**
    - User Query: {ctx.deps.user_query}\n
    - Current Date: {ctx.deps.current_date}\n
    - External Context: {ctx.deps.context}\n

    **Tone of the blog:**
    - Use semi-formal tone.
    - Do not use short form words like I've, I'll, I'm, etc.
    - You are allowed to use emojis and slangs.
    
   Think step by step.
    
    """
    return prompt

# Content Agent

class ContentAgentOutput(BaseModel):
    content: str = Field(description="The content of the section of the blog includind any sub-sections and paragraphs and tiles for the section")
    source_urls: list[str] = Field(description="The source urls of the blog")
    research_data: str = Field(description="The detailed research data of the blog gathered from the web_scraper tool, do not summarise or condense the data")

content_agent = Agent(
    model=model,
    tools=[Tool(get_source_url, takes_ctx=True), Tool(web_scraper, takes_ctx=True)],
    deps_type=State,
    result_type=ContentAgentOutput,
    instrument=True
)

@content_agent.system_prompt
async def get_content_agent_system_prompt(ctx: RunContext[State]):
    
    prompt = f"""
    You are a blog writer who writes the content of the blog by doing meticulous research and refereing to the sources, instructions and keypoints provided by the previous agent.

    Your goal is to write the content of the blog by doing meticulous research and refereing to the sources, instructions provided by the previous agent.

    **Instructions:**
    - Use the get_source_url tool to get the source urls for the blog. Note: Use this tool only once per section. Do  not use this more than 4 times.
    - Use the web_scraper tool to get the content of the urls.
    - Write the content of the section of the blog includind any sub-sections and paragraphs and tiles for the section.
    - If there is URLs in the content, use the web_scraper tool to get the content of the urls.

    Here are the instructions, keypoints, introduction and title provided by the previous agent:
    - User Query:\n {ctx.deps.user_query}
    - Title:\n {ctx.deps.blog_title}
    - External Context:\n {ctx.deps.context} \n
    - Research Data:\n {ctx.deps.research_data}
    - Previous Section Content:\n {ctx.deps.previous_section_content} \n
    Note: Previous section content is already generated, use it expand the make the content more consistent and coherent, in line with the previous section content.
    \n

    **Content of the blog:**
    - Make the content coherent and consistent with the previous section content.
    - Make the content engaging and interesting.
    - Make the content detailed and informative.
    - There can be multiple paragraphs in the content.
    - If the nature of the content is techincal or scientific, use the markdown compatible LaTeX to write the equations and formulas.
    
    **Tone of the blog:**
    - Use semi-formal tone.
    - Do not use short form words like I've, I'll, I'm, etc.
    - You are allowed to use emojis and slangs.

   Think step by step.
    
    """
    return prompt



# Editor Agent

@dataclass
class EditorAgentOutput(BaseModel):
    complete_blog: str = Field(default_factory=str, description="The complete blog")
    title: str = Field(description="The title of the blog")
    source_urls: list[str] = Field(description="The source urls of the blog")

class OutputResponse(BaseModel):
    complete_blog: str = Field(default="")
    title: str = Field(default="")
    source_urls: list[str] = Field(default_factory=list)


editor_agent = Agent(
    model=model,
    deps_type=State,
    result_type=EditorAgentOutput,
    instrument=True
)

@editor_agent.system_prompt
async def get_editor_agent_system_prompt(ctx: RunContext[State]):
    # Format blog content sections
    blog_content_formatted = ""
    for section, content in ctx.deps.blog_content.items():
        blog_content_formatted += f"\n{section}:\n{content}\n"

    prompt = f"""
    You are a blog editor who edits the blog based on the instructions and all the sections of the blog provided by the previous agent.

    Proof read the title, introduction, main content and conclusion provided by the previous agent.
    The compile everything into a one single consistent and coherent blog post.

    Quote the source urls in the blog.

    Here are the instructions, keypoints, title, sections and conclusion provided by the previous agent:
    - User Query:\n {ctx.deps.user_query}
    - ExternalContext:\n {ctx.deps.context}
    - Title:\n {ctx.deps.blog_title}
    - Blog Content:\n {blog_content_formatted}
    - Source Urls:\n {ctx.deps.source_urls}
    \n\n

    **Tone of the blog:**
    - Use semi-formal tone.
    - Do not use short form words like I've, I'll, I'm, etc.
    - You are allowed to use emojis and slangs.
    - If the nature of the content is techincal or scientific, use the LaTeX to write the equations and formulas.
    - Do not delete any information, only enhance and improve the content.

    NOTE: Make sure all the sections are present in the blog content, you can add more content to the sections if needed.

   Think step by step.

   """
    return prompt

# Graph Orchestration

@dataclass
class PlannerAgentNode(BaseNode[State]):
    """
    Planning the sections of the blog
    """
    async def run(self, ctx: GraphRunContext[State]) -> "ContentAgentNode":
        user_query = ctx.state.user_query
        context = ctx.state.context
        current_date = datetime.now().strftime("%Y-%m-%d")
        ctx.state.current_date = current_date
        response = await planner_agent.run(user_query, deps=ctx.state)
        response_data = response.data
        ctx.state.blog_title = response_data.title
        ctx.state.sections = response_data.sections
        ctx.state.instructions = response_data.instructions
        # for debugging
        print(f'\n\n Sections: {ctx.state.sections}\n\n')
        print(f'\n\n Title: {ctx.state.blog_title}\n\n')
        return ContentAgentNode()
    

@dataclass
class ContentAgentNode(BaseNode[State]):
    """
    Getting media insights from the user query
    """
    async def run(self, ctx: GraphRunContext[State]) -> "End":
        for section, instruction in zip(ctx.state.sections, ctx.state.instructions):
            ctx.state.url_tool_use_counter = 0
            ctx.state.web_scraper_tool_use_counter = 0
            query = f"For user query: {ctx.state.user_query}, write the content of the section: {section} with the instructions: {instruction}"
            response = await content_agent.run(query, deps=ctx.state)
            response_data = response.data
            ctx.state.blog_content[section] = response_data.content
            ctx.state.previous_section_content = response_data.content
            ctx.state.source_urls.extend(response_data.source_urls)
            ctx.state.research_data += '\n\n' + response_data.research_data

        return End(ctx.state)
    

'''
@dataclass
class EditorAgentNode(BaseNode[State]):
    """
    Editing the blog
    """
    async def run(self, ctx: GraphRunContext[State]) -> "End":
        user_query = ctx.state.user_query
        response = await editor_agent.run(user_query, deps=ctx.state)
        response_data = response.data
        ctx.state.complete_blog = response_data.complete_blog

        # for debugging
        print(f'\n\n Complete Blog: {ctx.state.complete_blog}\n\n')
        print(f'\n\n Title: {ctx.state.blog_title}\n\n')
        return End(response_data)
'''


def run_full_agent(user_query: str, user_id: str, context: str = ""):

    logfire.info(f"User {user_id} has run the agent with the prompt: {user_query}")

    current_date = datetime.now().strftime("%Y-%m-%d")
    state = State(user_query=user_query, current_date=current_date, context=context)
    graph = Graph(nodes=[PlannerAgentNode, ContentAgentNode])
    result = graph.run_sync(PlannerAgentNode(), state=state)
    result = result.output
    blog_content_formatted = ""
    for section, content in result.blog_content.items():
        blog_content_formatted += f"{content}\n"
    response = OutputResponse(
        complete_blog=blog_content_formatted,
        title=result.blog_title,
        source_urls=result.source_urls
    )
    print(response)
    return response

async def run_full_agent_async(user_query: str, user_id: str, context: str = ""):
    """Async version of run_full_agent that properly handles async operations"""
    
    logfire.info(f"User {user_id} has run the agent with the prompt: {user_query}")

    current_date = datetime.now().strftime("%Y-%m-%d")
    state = State(user_query=user_query, current_date=current_date, context=context)
    graph = Graph(nodes=[PlannerAgentNode, ContentAgentNode])
    result = await graph.run(PlannerAgentNode(), state=state)
    result = result.output
    blog_content_formatted = ""
    for section, content in result.blog_content.items():
        blog_content_formatted += f"\n{content}\n"
    response = OutputResponse(
        complete_blog=blog_content_formatted,
        title=result.blog_title,
        source_urls=result.source_urls
    )
    return response

async def main():
    user_prompt = "write a blog on special theory of relativity"
    user_id = "123"
    result = await run_full_agent_async(user_prompt, user_id)
    print('\n\n\n')
    print(result.complete_blog)
    print('\n\n\n')

if __name__ == "__main__":
    asyncio.run(main())
