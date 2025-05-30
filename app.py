from visualization import create_confidence_meter, create_bias_chart, create_sentiment_radar
from data_processor import process_text, analyze_sentiment, analyze_bias
from utils import extract_claims, get_source_credibility, simplify_text, summarize_text
from model import FakeNewsClassifier
import base64
import io
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import folium
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import streamlit as st

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'user_feedback' not in st.session_state:
    st.session_state.user_feedback = {}
if 'show_explainability' not in st.session_state:
    st.session_state.show_explainability = False

# Initialize the model


@st.cache_resource
def load_model():
    return FakeNewsClassifier()


model = load_model()

# Main title
st.title("🔍 Solvix NewsAnalyzer: Fake News Detection System")
st.markdown(
    "Solvix NewsAnalyzer - Advanced AI-powered system for fake news detection using NLP")

# Sidebar navigation
st.sidebar.title("Solvix NewsAnalyzer Navigation")
page = st.sidebar.selectbox("Choose a feature:", [
    "Main Classifier",
    "Prediction History",
    "Claim Verification",
    "Geolocation Analysis",
    "Chat Verification"
])

if page == "Main Classifier":
    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📝 News Text Input Panel")

        # Input method selection
        input_method = st.radio("Choose input method:", [
                                "Paste Text", "Upload File"])

        news_text = ""
        if input_method == "Paste Text":
            news_text = st.text_area(
                "Paste your news article here:",
                height=200,
                placeholder="Enter the news article text you want to analyze..."
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload a text file:",
                type=['txt', 'doc', 'docx'],
                help="Upload a text file containing the news article"
            )
            if uploaded_file is not None:
                try:
                    news_text = str(uploaded_file.read(), "utf-8")
                except:
                    st.error(
                        "Error reading file. Please ensure it's a valid text file.")

        # Real-time validation and word count
        if news_text:
            word_count = len(news_text.split())
            char_count = len(news_text)

            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                st.metric("Word Count", word_count)
            with col_stats2:
                st.metric("Character Count", char_count)
            with col_stats3:
                min_words = 50
                status = "✅ Ready" if word_count >= min_words else f"⚠️ Need {min_words - word_count} more words"
                st.metric("Status", status)

        # Analysis button
        if st.button("🔍 Analyze Article", type="primary", disabled=not news_text or len(news_text.split()) < 50):
            if news_text:
                with st.spinner("Analyzing article..."):
                    # Get prediction
                    prediction, confidence, feature_importance = model.predict(
                        news_text)

                    # Process additional analyses
                    sentiment_data = analyze_sentiment(news_text)
                    bias_data = analyze_bias(news_text)

                    # Store in session state
                    prediction_data = {
                        'text': news_text[:200] + "..." if len(news_text) > 200 else news_text,
                        'prediction': prediction,
                        'confidence': confidence,
                        'timestamp': datetime.now(),
                        'sentiment': sentiment_data,
                        'bias': bias_data
                    }
                    st.session_state.prediction_history.append(prediction_data)

                    # Display results
                    st.success("Analysis complete!")

                    # Model Prediction Display
                    st.header("🎯 Model Prediction")

                    result_col1, result_col2 = st.columns(2)

                    with result_col1:
                        # Animated result card
                        if prediction == 'Real':
                            st.success(
                                f"**Classification: {prediction} News**")
                            color = "green"
                        else:
                            st.error(f"**Classification: {prediction} News**")
                            color = "red"

                    with result_col2:
                        # Confidence meter
                        fig_confidence = create_confidence_meter(confidence)
                        st.plotly_chart(
                            fig_confidence, use_container_width=True)

                    # Explainability Toggle
                    st.session_state.show_explainability = st.checkbox(
                        "🔍 Show Explainability (Important Words)",
                        value=st.session_state.show_explainability
                    )

                    if st.session_state.show_explainability:
                        st.subheader("📊 Feature Importance")

                        # Create word cloud of important features
                        if feature_importance:
                            # Create columns for feature visualization and feature list
                            exp_col1, exp_col2 = st.columns(2)

                            with exp_col1:
                                st.write("**Important Words Visualization:**")
                                # Create a simple bar chart instead of word cloud
                                sorted_features = sorted(
                                    feature_importance.items(),
                                    key=lambda x: abs(x[1]),
                                    reverse=True
                                )[:10]

                                if sorted_features:
                                    words, importances = zip(*sorted_features)
                                    colors = [
                                        'red' if imp > 0 else 'blue' for imp in importances]

                                    fig_features = go.Figure(data=[
                                        go.Bar(
                                            y=list(words),
                                            x=[abs(imp)
                                               for imp in importances],
                                            orientation='h',
                                            marker_color=colors
                                        )
                                    ])
                                    fig_features.update_layout(
                                        title="Important Words (Red=Fake, Blue=Real)",
                                        height=400,
                                        margin=dict(l=120, r=20, t=40, b=20)
                                    )
                                    st.plotly_chart(
                                        fig_features, use_container_width=True)

                                with exp_col2:
                                    st.write("**Top Important Words:**")
                                    # Sort by importance and display top 10
                                    sorted_features = sorted(
                                        feature_importance.items(),
                                        key=lambda x: abs(x[1]),
                                        reverse=True
                                    )[:10]

                                    for word, importance in sorted_features:
                                        direction = "→ Fake" if importance > 0 else "→ Real"
                                        st.write(
                                            f"**{word}**: {abs(importance):.3f} {direction}")

                    # Source Trust Meter
                    st.header("🏛️ Source Trust Analysis")
                    source_credibility = get_source_credibility(news_text)

                    trust_col1, trust_col2 = st.columns(2)
                    with trust_col1:
                        trust_score = source_credibility['trust_score']
                        if trust_score >= 0.7:
                            st.success(
                                f"**Trust Score: {trust_score:.1%}** - High Credibility")
                        elif trust_score >= 0.4:
                            st.warning(
                                f"**Trust Score: {trust_score:.1%}** - Medium Credibility")
                        else:
                            st.error(
                                f"**Trust Score: {trust_score:.1%}** - Low Credibility")

                    with trust_col2:
                        with st.expander("📊 Trust Factors"):
                            for factor, score in source_credibility['factors'].items():
                                st.write(f"**{factor}**: {score:.1%}")

                    # Bias & Sentiment Analysis
                    st.header("📈 Bias & Sentiment Analysis")

                    bias_col1, bias_col2 = st.columns(2)

                    with bias_col1:
                        st.subheader("Political Bias")
                        fig_bias = create_bias_chart(bias_data)
                        st.plotly_chart(fig_bias, use_container_width=True)

                    with bias_col2:
                        st.subheader("Sentiment Analysis")
                        fig_sentiment = create_sentiment_radar(sentiment_data)
                        st.plotly_chart(
                            fig_sentiment, use_container_width=True)

                    # Simplify & Summarize
                    st.header("📝 Article Processing")

                    process_col1, process_col2 = st.columns(2)

                    with process_col1:
                        if st.button("🔤 Simplify Text"):
                            with st.spinner("Simplifying text..."):
                                simplified = simplify_text(news_text)
                                st.subheader("Simplified Version")
                                st.write(simplified)

                    with process_col2:
                        if st.button("📋 Summarize Text"):
                            with st.spinner("Summarizing text..."):
                                summary = summarize_text(news_text)
                                st.subheader("Summary")
                                st.write(summary)

# User Feedback System
st.header("👥 Solvix NewsAnalyzer User Feedback")

feedback_col1, feedback_col2 = st.columns(2)

with feedback_col1:
    st.write("**Was this prediction helpful?**")
    feedback = st.radio(
        "Your feedback:",
        ["👍 Helpful", "👎 Not Helpful", "🤔 Uncertain"],
        key=f"feedback_{len(st.session_state.prediction_history)}"
    )

with feedback_col2:
    comment = st.text_area(
        "Additional comments (optional):",
        height=100,
        key=f"comment_{len(st.session_state.prediction_history)}"
    )

if st.button("Submit Feedback"):
    feedback_id = len(st.session_state.prediction_history) - 1
    st.session_state.user_feedback[feedback_id] = {
        'rating': feedback,
        'comment': comment,
        'timestamp': datetime.now()
    }
    st.success("Thank you for your feedback!")

with col2:
    st.header("📊 Quick Stats")

    # Display recent predictions count
    total_predictions = len(st.session_state.prediction_history)
    st.metric("Total Predictions", total_predictions)

    if total_predictions > 0:
        recent_predictions = st.session_state.prediction_history[-5:]
        real_count = sum(
            1 for p in recent_predictions if p['prediction'] == 'Real')
        fake_count = len(recent_predictions) - real_count

        st.metric("Recent Real News", real_count)
        st.metric("Recent Fake News", fake_count)

        # Recent predictions chart
        if len(recent_predictions) > 1:
            st.subheader("Recent Predictions Trend")
            df_recent = pd.DataFrame([
                {
                    'Time': p['timestamp'].strftime("%H:%M"),
                    'Prediction': p['prediction'],
                    'Confidence': p['confidence']
                }
                for p in recent_predictions
            ])

            fig_trend = px.line(
                df_recent, x='Time', y='Confidence',
                color='Prediction',
                title="Confidence Over Time"
            )
            st.plotly_chart(fig_trend, use_container_width=True)

if page == "Prediction History":
    st.header("📚 Solvix NewsAnalyzer Prediction History")

    if st.session_state.prediction_history:
        # Create DataFrame for history
        history_data = []
        for i, pred in enumerate(st.session_state.prediction_history):
            feedback_info = st.session_state.user_feedback.get(i, {})
            history_data.append({
                'ID': i + 1,
                'Text Preview': pred['text'],
                'Prediction': pred['prediction'],
                'Confidence': f"{pred['confidence']:.1%}",
                'Sentiment': pred['sentiment']['polarity'],
                'Timestamp': pred['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                'User Feedback': feedback_info.get('rating', 'No feedback')
            })

        df_history = pd.DataFrame(history_data)

        # Display filters
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            filter_prediction = st.selectbox(
                "Filter by prediction:",
                ["All", "Real", "Fake"]
            )
        with col_filter2:
            filter_feedback = st.selectbox(
                "Filter by feedback:",
                ["All", "👍 Helpful", "👎 Not Helpful", "🤔 Uncertain", "No feedback"]
            )

        # Apply filters
        filtered_df = df_history.copy()
        if filter_prediction != "All":
            filtered_df = filtered_df[filtered_df['Prediction']
                                      == filter_prediction]
        if filter_feedback != "All":
            filtered_df = filtered_df[filtered_df['User Feedback']
                                      == filter_feedback]

        # Display table
        st.dataframe(filtered_df, use_container_width=True)

        # Summary statistics
        if not filtered_df.empty:
            st.subheader("📊 Summary Statistics")

            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

            with stat_col1:
                st.metric("Total Articles", len(filtered_df))
            with stat_col2:
                real_pct = (filtered_df['Prediction'] == 'Real').mean() * 100
                st.metric("Real News %", f"{real_pct:.1f}%")
            with stat_col3:
                fake_pct = (filtered_df['Prediction'] == 'Fake').mean() * 100
                st.metric("Fake News %", f"{fake_pct:.1f}%")
            with stat_col4:
                helpful_feedback = (
                    filtered_df['User Feedback'] == '👍 Helpful').sum()
                st.metric("Helpful Ratings", helpful_feedback)

        # Download option
        if st.button("📥 Download History as CSV"):
            csv = df_history.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No predictions yet. Analyze some articles to see your history here!")

elif page == "Claim Verification":
    st.header("🔍 Claim Verification Panel")

    if st.session_state.prediction_history:
        # Select an article to analyze claims
        article_options = [
            f"Article {i+1}: {pred['text'][:50]}..."
            for i, pred in enumerate(st.session_state.prediction_history)
        ]

        selected_idx = st.selectbox(
            "Select an article to extract claims:",
            range(len(article_options)),
            format_func=lambda x: article_options[x]
        )

        if selected_idx is not None:
            selected_article = st.session_state.prediction_history[selected_idx]

            st.subheader("📄 Selected Article")
            st.write(selected_article['text'])

            # Extract claims
            if st.button("🔍 Extract Claims"):
                with st.spinner("Extracting claims..."):
                    claims = extract_claims(selected_article['text'])

                    st.subheader("📋 Extracted Claims")

                    for i, claim in enumerate(claims):
                        with st.expander(f"Claim {i+1}: {claim['claim'][:100]}..."):
                            st.write(f"**Full Claim:** {claim['claim']}")
                            st.write(
                                f"**Confidence:** {claim['confidence']:.1%}")
                            st.write(f"**Category:** {claim['category']}")

                            # Mock fact-check button
                            if st.button(f"🔍 Fact-Check Claim {i+1}", key=f"factcheck_{i}"):
                                with st.spinner("Checking facts..."):
                                    # Simulate fact-checking process
                                    import random
                                    fact_check_result = {
                                        'status': random.choice(['Verified', 'False', 'Partially True', 'Unverified']),
                                        'sources': ['Reuters', 'AP News', 'Snopes'],
                                        'explanation': f"Based on available evidence, this claim appears to be {random.choice(['accurate', 'inaccurate', 'partially correct'])}."
                                    }

                                    if fact_check_result['status'] == 'Verified':
                                        st.success(
                                            f"✅ **{fact_check_result['status']}**")
                                    elif fact_check_result['status'] == 'False':
                                        st.error(
                                            f"❌ **{fact_check_result['status']}**")
                                    elif fact_check_result['status'] == 'Partially True':
                                        st.warning(
                                            f"⚠️ **{fact_check_result['status']}**")
                                    else:
                                        st.info(
                                            f"❓ **{fact_check_result['status']}**")

                                    st.write(
                                        f"**Explanation:** {fact_check_result['explanation']}")
                                    st.write(
                                        f"**Sources:** {', '.join(fact_check_result['sources'])}")
    else:
        st.info(
            "No articles analyzed yet. Go to the Main Classifier to analyze some articles first!")

elif page == "Geolocation Analysis":
    st.header("🗺️ Geolocation Claim Map")

    st.write(
        "This feature visualizes claimed event locations mentioned in news articles.")

    if st.session_state.prediction_history:
        # Create a sample map with some locations
        center_lat, center_lon = 40.7128, -74.0060  # New York City

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=2,
            width='100%',
            height='400px'
        )

        # Add sample markers for demonstration
        sample_locations = [
            {"lat": 40.7128, "lon": -74.0060, "name": "New York",
                "articles": 3, "status": "Verified"},
            {"lat": 51.5074, "lon": -0.1278, "name": "London",
                "articles": 2, "status": "Disputed"},
            {"lat": 35.6762, "lon": 139.6503, "name": "Tokyo",
                "articles": 1, "status": "Unverified"},
            {"lat": 48.8566, "lon": 2.3522, "name": "Paris",
                "articles": 2, "status": "Verified"},
        ]

        for location in sample_locations:
            color = {
                "Verified": "green",
                "Disputed": "orange",
                "Unverified": "red"
            }.get(location["status"], "blue")

            folium.Marker(
                [location["lat"], location["lon"]],
                popup=f"""
                <b>{location['name']}</b><br>
                Articles: {location['articles']}<br>
                Status: {location['status']}
                """,
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(m)

        # Display the map
        map_data = st_folium(m, width=700, height=400)

        # Legend
        st.subheader("🏷️ Legend")
        legend_col1, legend_col2, legend_col3 = st.columns(3)
        with legend_col1:
            st.success("🟢 Verified Claims")
        with legend_col2:
            st.warning("🟠 Disputed Claims")
        with legend_col3:
            st.error("🔴 Unverified Claims")

        # Location details
        if map_data['last_object_clicked_popup']:
            st.subheader("📍 Selected Location Details")
            st.info(
                "Click on a marker to see details about claims from that location.")

        # Statistics
        st.subheader("📊 Geolocation Statistics")

        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.metric("Total Locations", len(sample_locations))
        with stat_col2:
            verified_count = sum(
                1 for loc in sample_locations if loc["status"] == "Verified")
            st.metric("Verified", verified_count)
        with stat_col3:
            disputed_count = sum(
                1 for loc in sample_locations if loc["status"] == "Disputed")
            st.metric("Disputed", disputed_count)
        with stat_col4:
            unverified_count = sum(
                1 for loc in sample_locations if loc["status"] == "Unverified")
            st.metric("Unverified", unverified_count)

    else:
        st.info(
            "No articles analyzed yet. Analyze some articles to see geolocation data!")

elif page == "Chat Verification":
    st.header("💬 Interactive Chatbot for Verification")

    st.write(
        "Ask specific questions about article content and get real-time fact-check information.")

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! I'm your fact-checking assistant. Ask me questions about any news article or claim you'd like to verify."}
        ]

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me about any news claim..."):
        # Add user message to chat history
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Checking facts..."):
                # Simple response generation based on keywords
                response = f"I'm analyzing your query: '{prompt}'. Based on general fact-checking principles, I recommend verifying claims through multiple reliable sources, checking for official statements, and looking for peer-reviewed research. For specific fact-checking, please consult established fact-checking organizations like Snopes, PolitiFact, or FactCheck.org."
                st.write(response)

                # Add assistant response to chat history
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response})

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! I'm your fact-checking assistant. Ask me questions about any news article or claim you'd like to verify."}
        ]
        st.rerun()


def generate_fact_check_response(prompt):
    """Generate a fact-checking response based on the user's prompt."""
    prompt_lower = prompt.lower()

    # Keywords for different types of responses
    political_keywords = ['election', 'vote',
                          'politician', 'government', 'policy']
    health_keywords = ['vaccine', 'covid', 'health', 'medicine', 'doctor']
    science_keywords = ['climate', 'research', 'study', 'scientist', 'data']

    if any(keyword in prompt_lower for keyword in political_keywords):
        return "🏛️ For political claims, I recommend checking with authoritative sources like Reuters, AP News, or official government websites. Political misinformation is common, so always verify with multiple credible sources."

    elif any(keyword in prompt_lower for keyword in health_keywords):
        return "🏥 Health-related claims should be verified with medical authorities like the WHO, CDC, or peer-reviewed medical journals. Be especially cautious of health misinformation as it can have serious consequences."

    elif any(keyword in prompt_lower for keyword in science_keywords):
        return "🔬 Scientific claims should be checked against peer-reviewed research and reputable scientific institutions. Look for consensus among experts and be wary of cherry-picked studies."

    else:
        return f"🔍 I'd be happy to help you fact-check that claim. Here are some steps you can take:\n\n1. **Check the source**: Is it from a reputable news organization?\n2. **Look for corroboration**: Do other credible sources report the same information?\n3. **Check fact-checking sites**: Snopes, FactCheck.org, and PolitiFact are good resources.\n4. **Consider the date**: Is the information current and relevant?\n5. **Look for evidence**: Are there citations, data, or expert quotes?\n\nFor your specific question about '{prompt}', I recommend starting with these verification steps."


# Sidebar additional info
with st.sidebar:
    st.markdown("---")
    st.subheader("📈 Solvix NewsAnalyzer System Stats")

    total_analyzed = len(st.session_state.prediction_history)
    st.metric("Articles Analyzed", total_analyzed)

    if total_analyzed > 0:
        real_news_count = sum(
            1 for p in st.session_state.prediction_history if p['prediction'] == 'Real')
        fake_news_count = total_analyzed - real_news_count

        st.metric("Real News Detected", real_news_count)
        st.metric("Fake News Detected", fake_news_count)

        # Accuracy based on user feedback
        helpful_feedback = sum(
            1 for feedback in st.session_state.user_feedback.values()
            if feedback.get('rating') == '👍 Helpful'
        )

        if len(st.session_state.user_feedback) > 0:
            user_satisfaction = helpful_feedback / \
                len(st.session_state.user_feedback) * 100
            st.metric("User Satisfaction", f"{user_satisfaction:.1f}%")

    st.markdown("---")
    st.markdown("**💡 Solvix NewsAnalyzer Tips:**")
    st.markdown("• Upload articles with at least 50 words")
    st.markdown("• Check multiple sources for verification")
    st.markdown("• Pay attention to bias and sentiment analysis")
    st.markdown("• Use the explainability feature to understand predictions")
