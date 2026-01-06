import plotly.graph_objects as go
import plotly.express as px

# Data from the provided JSON
quality_settings = ["Strict", "Standard", "Relaxed", "No Filter"]
frame_retention = [45, 74, 89, 100]
quality_score = [95, 85, 70, 60]
time_reduction = [60, 35, 15, 0]

# Create grouped bar chart
fig = go.Figure()

# Add bars for each metric using the brand colors
fig.add_trace(go.Bar(
    name='Frame Retention',
    x=quality_settings,
    y=frame_retention,
    marker_color='#1FB8CD',
    text=[f'{val}%' for val in frame_retention],
    textposition='outside'
))

fig.add_trace(go.Bar(
    name='Quality Score',
    x=quality_settings,
    y=quality_score,
    marker_color='#DB4545',
    text=[f'{val}%' for val in quality_score],
    textposition='outside'
))

fig.add_trace(go.Bar(
    name='Time Reduction',
    x=quality_settings,
    y=time_reduction,
    marker_color='#2E8B57',
    text=[f'{val}%' for val in time_reduction],
    textposition='outside'
))

# Update layout
fig.update_layout(
    title='Quality Filter Impact on Frame Retention',
    xaxis_title='Quality Setting',
    yaxis_title='Percentage (%)',
    barmode='group',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.05,
        xanchor='center',
        x=0.5
    )
)

# Update traces for better visibility
fig.update_traces(cliponaxis=False)

# Save as both PNG and SVG
fig.write_image('quality_filter_chart.png')
fig.write_image('quality_filter_chart.svg', format='svg')

fig.show()