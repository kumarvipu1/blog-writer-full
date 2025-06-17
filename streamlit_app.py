import streamlit as st
import json
from datetime import datetime
#from blog_agent import run_full_agent
from auto_blog_agent import run_full_agent

# Initialize session state
if 'chat_memory' not in st.session_state:
    st.session_state.chat_memory = []
if 'blog_results' not in st.session_state:
    st.session_state.blog_results = []

def store_blog_info(user_query, result):
    """Store blog information in chat memory"""
    blog_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_query": user_query,
        "blog_title": result.title,
        "complete_blog": result.complete_blog,
        "source_urls": result.source_urls
    }
    
    # Store as string in chat memory
    memory_string = f"""
    Query: {user_query}
    Title: {result.title}
    Blog Content: {result.complete_blog}
    Source URLs: {', '.join(result.source_urls)}
    Generated at: {blog_info['timestamp']}
    """
    
    st.session_state.chat_memory.append(memory_string)
    st.session_state.blog_results.append(blog_info)

def main():
    st.set_page_config(
        page_title="AI Blog Writer",
        page_icon="✍️",
        layout="wide"
    )
    
    st.title("🤖 AI Documentation Writer")
    st.markdown("Generate well-researched documentation on any topic!")
    
    # Sidebar for chat memory
    with st.sidebar:
        st.header("📝 Blog History")
        if st.session_state.blog_results:
            for i, blog_info in enumerate(st.session_state.blog_results):
                with st.expander(f"Blog {i+1}: {blog_info['blog_title'][:50]}..."):
                    st.write(f"**Query:** {blog_info['user_query']}")
                    st.write(f"**Generated:** {blog_info['timestamp']}")
                    if st.button(f"View Blog {i+1}", key=f"view_{i}"):
                        st.session_state.selected_blog = i
        
        if st.button("Clear History"):
            st.session_state.chat_memory = []
            st.session_state.blog_results = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Generate New Documentation")
        
        # User input
        user_query = st.text_area(
            "Enter your blog topic or question:",
            placeholder="e.g., Write a blog about the impact of artificial intelligence on healthcare",
            height=100
        )
        
        # Add context input
        context = st.text_area(
            "Additional Context (Optional):",
            placeholder="Add any specific context, requirements, or guidelines for the blog...",
            height=150
        )
        
        generate_button = st.button("🚀 Generate Blog", type="primary")
        
        if generate_button and user_query:
            with st.spinner("🔍 Researching and writing your blog... This may take a few minutes."):
                try:
                    # Use default user ID
                    default_user_id = "streamlit_user"
                    
                    # Call the blog agent with context
                    result = run_full_agent(user_query, default_user_id, context=context)
                    
                    # Store in memory
                    store_blog_info(user_query, result)
                    
                    # Display success message
                    st.success("✅ Blog generated successfully!")
                    
                    # Display the complete blog
                    st.header("📖 Generated Blog")
                    st.markdown(result.complete_blog)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Blog",
                        data=result.complete_blog,
                        file_name=f"blog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error generating blog: {str(e)}")
        
        elif generate_button and not user_query:
            st.warning("⚠️ Please enter a blog topic or question.")
    
    with col2:
        st.header("💬 Follow-up Questions")
        
        if st.session_state.blog_results:
            st.info("💡 You can ask follow-up questions about your generated blogs!")
            
            followup_query = st.text_area(
                "Ask a follow-up question:",
                placeholder="e.g., Can you expand on the first section?",
                height=80
            )
            
            if st.button("🔄 Generate Follow-up"):
                if followup_query:
                    # Get the latest blog context
                    latest_blog = st.session_state.chat_memory[-1] if st.session_state.chat_memory else ""
                    
                    # Combine context with follow-up query
                    enhanced_query = f"""
                    Based on the previous blog:
                    {latest_blog}
                    
                    Follow-up question: {followup_query}
                    """
                    
                    with st.spinner("🔍 Processing follow-up..."):
                        try:
                            result = run_full_agent(enhanced_query, "streamlit_user")
                            store_blog_info(followup_query, result)
                            st.success("✅ Follow-up generated!")
                            st.markdown(result.complete_blog)
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        else:
            st.info("Generate a blog first to ask follow-up questions!")
    
    # Display selected blog from sidebar
    if hasattr(st.session_state, 'selected_blog'):
        selected_index = st.session_state.selected_blog
        if 0 <= selected_index < len(st.session_state.blog_results):
            selected_blog = st.session_state.blog_results[selected_index]
            
            st.header(f"📖 {selected_blog['blog_title']}")
            st.markdown(selected_blog['complete_blog'])
            
            # Clear selection
            if st.button("Close Blog View"):
                del st.session_state.selected_blog
                st.rerun()

if __name__ == "__main__":
    main() 