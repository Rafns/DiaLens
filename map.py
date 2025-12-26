# FUNGSI: BUAT PETA GLOBAL DIABETES
import pandas as pd 
import plotly.graph_objects as go
import country_converter as coco
import streamlit as st

def make_diabetes_map():
    try:
        # Baca file CSV
        df = pd.read_csv('diabetes_world_data.csv')
        
        # Hapus baris kosong
        df = df.dropna(subset=['Location', 'Value'])
        
        # Konversi nama negara ke iso_alpha
        df['iso_alpha'] = coco.convert(names=df['Location'], to='ISO3', not_found='None')
        df = df[df['iso_alpha'] != 'None']
        
        # Konversi Value ke numerik
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df = df.dropna(subset=['Value'])

        # Buat kategori warna
        bins = [0, 100, 500, 1000, 10000, 20000, float('inf')]
        labels = ['<100 thousand', '100-500 thousand', '500 thousand-1 million', '1-10 million', '10-20 million', '>20 million']
        df['Category'] = pd.cut(df['Value'], bins=bins, labels=labels, right=False)

        # Mapping warna
        color_map = {
            '<100 thousand': '#457B9D',           # Biru muda
            '100-500 thousand': '#1D3557',         # Biru tua
            '500 thousand-1 million': '#000000',   # Hitam
            '1-10 million': '#E9C46D',             # Pink muda
            '10-20 million': '#F4A261',            # Pink tua
            '>20 million': '#E76F51'               # Merah
        }

        df['Color'] = df['Category'].map(color_map)

        # Buat peta choropleth
        fig = go.Figure(data=go.Choropleth(
            locations=df['iso_alpha'],
            z=df['Value'],  # Untuk hover info
            colorscale=[
                [0, '#457B9D'],       # <100k
                [0.166, '#457B9D'],
                [0.166, '#1D3557'],   # 100k-500k
                [0.333, '#1D3557'],
                [0.333, '#000000'],   # 500k-1M
                [0.5, '#000000'],
                [0.5, '#E9C46D'],     # 1-10M
                [0.666, '#E9C46D'],
                [0.666, '#F4A261'],   # 10-20M
                [0.833, '#F4A261'],
                [0.833, '#E76F51'],   # >20M
                [1, '#E76F51']
            ],
            zmin=0,
            zmax=20000,  # batas atas untuk warna >20M
            text=df['Location'],
            hovertemplate="<b>%{text}</b><br>Cases: %{z:,} thousand<br><extra></extra>",
            marker_line_color='darkgray',
            showscale=False,
        ))

        # Tambahkan legenda manual (karena Plotly tidak punya legenda kategori)
        fig.add_trace(go.Scattergeo(
            lon=[0], lat=[0],
            mode='markers',
            marker=dict(size=0),
            showlegend=True,
            name='<100 thousand'
        ))
        fig.add_trace(go.Scattergeo(
            lon=[0], lat=[0],
            mode='markers',
            marker=dict(size=0),
            showlegend=True,
            name='100-500 thousand'
        ))
        fig.add_trace(go.Scattergeo(
            lon=[0], lat=[0],
            mode='markers',
            marker=dict(size=0),
            showlegend=True,
            name='500 thousand-1 million'
        ))
        fig.add_trace(go.Scattergeo(
            lon=[0], lat=[0],
            mode='markers',
            marker=dict(size=0),
            showlegend=True,
            name='1-10 million'
        ))
        fig.add_trace(go.Scattergeo(
            lon=[0], lat=[0],
            mode='markers',
            marker=dict(size=0),
            showlegend=True,
            name='10-20 million'
        ))
        fig.add_trace(go.Scattergeo(
            lon=[0], lat=[0],
            mode='markers',
            marker=dict(size=0),
            showlegend=True,
            name='>20 million'
        ))

        # Update layout
        fig.update_layout(
            title_text='Estimated Number of Adults (20–79) with Diabetes in 2024',
            geo=dict(
                showframe=False,
                showcoastlines=False,
                projection_type='equirectangular',
                bgcolor='rgba(0,0,0,0)'
            ),
            margin=dict(l=0, r=0, t=60, b=0),
            annotations=[dict(
                x=0.95,
                y=0.02,
                xref='paper',
                yref='paper',
                text='Source: <a href="https://diabetesatlas.org/data/en/world/">IDF Diabetes Atlas</a>',
                showarrow=False,
                font=dict(size=10)
            )],
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(0,0,0,0.8)',
                bordercolor='black',
                borderwidth=1
            )
        )

        return fig

    except Exception as e:
        st.warning(f"⚠️ Gagal memuat peta global: {str(e)}")
        return None
