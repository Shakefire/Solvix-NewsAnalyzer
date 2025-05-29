import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

def create_confidence_meter(confidence):
    """Create a confidence meter visualization."""
    # Create a gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "gray"},
                {'range': [50, 75], 'color': "lightblue"},
                {'range': [75, 100], 'color': "blue"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_bias_chart(bias_data):
    """Create a political bias visualization."""
    bias_score = bias_data.get('political_bias', 0.0)
    emotional_intensity = bias_data.get('emotional_intensity', 0.0)
    
    # Create a horizontal bar chart for bias
    fig = go.Figure()
    
    # Political bias bar
    color = 'red' if bias_score > 0 else 'blue' if bias_score < 0 else 'gray'
    
    fig.add_trace(go.Bar(
        x=[abs(bias_score)],
        y=['Political Bias'],
        orientation='h',
        marker_color=color,
        name='Political Bias',
        text=[f"{bias_score:.2f}"],
        textposition='inside'
    ))
    
    # Emotional intensity bar
    fig.add_trace(go.Bar(
        x=[emotional_intensity],
        y=['Emotional Intensity'],
        orientation='h',
        marker_color='orange',
        name='Emotional Intensity',
        text=[f"{emotional_intensity:.2f}"],
        textposition='inside'
    ))
    
    fig.update_layout(
        title="Bias Analysis",
        xaxis_title="Intensity",
        yaxis_title="Bias Type",
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    return fig

def create_sentiment_radar(sentiment_data):
    """Create a radar chart for sentiment analysis."""
    emotions = sentiment_data.get('emotions', {'positive': 0, 'negative': 0, 'neutral': 1})
    
    categories = list(emotions.keys())
    values = list(emotions.values())
    
    # Close the radar chart
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Sentiment',
        line_color='rgb(32, 201, 151)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Sentiment Analysis",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_feature_importance_chart(feature_importance, top_n=10):
    """Create a feature importance visualization."""
    if not feature_importance:
        return go.Figure()
    
    # Sort features by absolute importance
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_n]
    
    words, importances = zip(*sorted_features)
    
    # Create color based on positive/negative importance
    colors = ['red' if imp > 0 else 'blue' for imp in importances]
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(importances),
            y=list(words),
            orientation='h',
            marker_color=colors,
            text=[f"{imp:.3f}" for imp in importances],
            textposition='inside'
        )
    ])
    
    fig.update_layout(
        title="Feature Importance (Red = Fake indicators, Blue = Real indicators)",
        xaxis_title="Importance Score",
        yaxis_title="Words",
        height=400,
        margin=dict(l=100, r=20, t=40, b=20)
    )
    
    return fig

def create_prediction_timeline(predictions_history):
    """Create a timeline of predictions."""
    if not predictions_history:
        return go.Figure()
    
    df = pd.DataFrame([
        {
            'timestamp': pred['timestamp'],
            'prediction': pred['prediction'],
            'confidence': pred['confidence']
        }
        for pred in predictions_history
    ])
    
    fig = px.scatter(
        df, 
        x='timestamp', 
        y='confidence',
        color='prediction',
        title='Prediction Timeline',
        color_discrete_map={'Real': 'blue', 'Fake': 'red'}
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_text_statistics_chart(text_stats):
    """Create visualization for text statistics."""
    if not text_stats:
        return go.Figure()
    
    basic_stats = text_stats.get('basic_stats', {})
    
    stats_names = ['Sentences', 'Words', 'Avg Sentence Length', 'Avg Word Length', 'Vocabulary Diversity']
    stats_values = [
        basic_stats.get('sentence_count', 0),
        basic_stats.get('word_count', 0),
        basic_stats.get('avg_sentence_length', 0),
        basic_stats.get('avg_word_length', 0),
        basic_stats.get('vocabulary_diversity', 0) * 100  # Convert to percentage
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=stats_names,
            y=stats_values,
            marker_color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
        )
    ])
    
    fig.update_layout(
        title="Text Statistics",
        xaxis_title="Metrics",
        yaxis_title="Values",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_word_frequency_chart(word_frequencies, top_n=15):
    """Create a word frequency bar chart."""
    if not word_frequencies:
        return go.Figure()
    
    # Take top N words
    top_words = word_frequencies[:top_n]
    words, frequencies = zip(*top_words) if top_words else ([], [])
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(frequencies),
            y=list(words),
            orientation='h',
            marker_color='lightblue'
        )
    ])
    
    fig.update_layout(
        title="Most Frequent Words",
        xaxis_title="Frequency",
        yaxis_title="Words",
        height=400,
        margin=dict(l=100, r=20, t=40, b=20)
    )
    
    return fig

def create_comparison_chart(real_confidence, fake_confidence):
    """Create a comparison chart between real and fake confidence."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Real News Confidence',
        x=['Confidence'],
        y=[real_confidence * 100],
        marker_color='blue'
    ))
    
    fig.add_trace(go.Bar(
        name='Fake News Confidence',
        x=['Confidence'],
        y=[fake_confidence * 100],
        marker_color='red'
    ))
    
    fig.update_layout(
        title='Real vs Fake News Confidence',
        yaxis_title='Confidence (%)',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        barmode='group'
    )
    
    return fig

def create_trust_meter(trust_score):
    """Create a trust meter similar to confidence meter."""
    colors = ["red", "orange", "yellow", "lightgreen", "green"]
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = trust_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Source Trust Score"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 20], 'color': "red"},
                {'range': [20, 40], 'color': "orange"},
                {'range': [40, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "lightgreen"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig
