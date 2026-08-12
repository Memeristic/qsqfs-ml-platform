"""Blue / gold / white theme for the Streamlit interface.

Colours are chosen to match the publication figures, so a chart in the app and
the same chart in a thesis look like one family. Gold is reserved for emphasis
and headline metrics; blue carries structure; white keeps it readable.
"""

NAVY = "#0B3C6B"
BLUE = "#1F6FB2"
LIGHT_BLUE = "#7FB3DC"
PALE_BLUE = "#F4F8FC"
GOLD = "#C9A227"
DARK_GOLD = "#8C6F14"
PALE_GOLD = "#FBF6E3"
WHITE = "#FFFFFF"
INK = "#0F172A"
MUTED = "#64748B"
GOOD = "#177245"
WARN = "#B45309"
BAD = "#9B1C1C"

CSS = f"""
<style>
  :root {{
    --navy:{NAVY}; --blue:{BLUE}; --lblue:{LIGHT_BLUE}; --pblue:{PALE_BLUE};
    --gold:{GOLD}; --dgold:{DARK_GOLD}; --pgold:{PALE_GOLD};
    --ink:{INK}; --muted:{MUTED};
  }}

  .stApp {{ background: {WHITE}; }}
  .block-container {{ padding-top: 2rem; max-width: 1400px; }}

  /* ---- header band ---- */
  .qsq-hero {{
    background: linear-gradient(120deg, {NAVY} 0%, {BLUE} 65%, #2E86C9 100%);
    border-radius: 14px; padding: 1.4rem 1.7rem; margin-bottom: 1.2rem;
    border-bottom: 4px solid {GOLD};
    box-shadow: 0 6px 22px rgba(11,60,107,.18);
  }}
  .qsq-hero h1 {{
    color:{WHITE}; margin:0; font-size:1.85rem; font-weight:700;
    letter-spacing:-.02em;
  }}
  .qsq-hero p {{ color:#D8E8F6; margin:.35rem 0 0; font-size:.93rem; }}
  .qsq-hero .accent {{ color:{GOLD}; font-weight:700; }}

  /* ---- section headings ---- */
  .qsq-section {{
    display:flex; align-items:center; gap:.6rem; margin:1.4rem 0 .7rem;
    padding-bottom:.4rem; border-bottom:2px solid {PALE_BLUE};
  }}
  .qsq-section .bar {{
    width:5px; height:22px; background:{GOLD}; border-radius:3px;
  }}
  .qsq-section h3 {{ margin:0; color:{NAVY}; font-size:1.12rem; font-weight:650; }}

  /* ---- metric cards ---- */
  div[data-testid="stMetric"] {{
    background:{PALE_BLUE}; border:1px solid #DCE9F5; border-left:4px solid {BLUE};
    border-radius:10px; padding:.85rem 1rem;
  }}
  div[data-testid="stMetric"] label p {{
    color:{MUTED}!important; font-size:.76rem!important;
    text-transform:uppercase; letter-spacing:.05em; font-weight:600;
  }}
  div[data-testid="stMetricValue"] {{
    color:{NAVY}!important; font-size:1.5rem!important; font-weight:700;
  }}

  /* ---- buttons ---- */
  .stButton > button {{
    background:{BLUE}; color:{WHITE}; border:none; border-radius:9px;
    padding:.55rem 1.15rem; font-weight:600; transition:all .16s ease;
    box-shadow:0 2px 6px rgba(31,111,178,.24);
  }}
  .stButton > button:hover {{
    background:{NAVY}; box-shadow:0 4px 14px rgba(11,60,107,.32);
    transform:translateY(-1px);
  }}
  .stButton > button[kind="primary"] {{
    background:linear-gradient(100deg,{GOLD} 0%,#D9B43A 100%);
    color:{NAVY}; font-weight:700; box-shadow:0 3px 10px rgba(201,162,39,.36);
  }}
  .stButton > button[kind="primary"]:hover {{
    background:linear-gradient(100deg,{DARK_GOLD} 0%,{GOLD} 100%); color:{WHITE};
  }}
  .stDownloadButton > button {{
    background:{PALE_GOLD}; color:{DARK_GOLD}; border:2px solid {GOLD};
    border-radius:9px; font-weight:650;
  }}
  .stDownloadButton > button:hover {{ background:{GOLD}; color:{WHITE}; }}

  /* ---- sidebar ---- */
  section[data-testid="stSidebar"] {{
    background:linear-gradient(180deg,{PALE_BLUE} 0%,{WHITE} 320px);
    border-right:1px solid #E2ECF6;
  }}
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 {{
    color:{NAVY}; font-size:.95rem; font-weight:700;
    text-transform:uppercase; letter-spacing:.045em;
    border-left:4px solid {GOLD}; padding-left:.55rem; margin-top:1.1rem;
  }}

  /* ---- tabs ---- */
  .stTabs [data-baseweb="tab-list"] {{ gap:2px; border-bottom:2px solid {PALE_BLUE}; }}
  .stTabs [data-baseweb="tab"] {{
    background:transparent; color:{MUTED}; font-weight:600;
    border-radius:8px 8px 0 0; padding:.5rem 1rem;
  }}
  .stTabs [aria-selected="true"] {{
    background:{PALE_BLUE}!important; color:{NAVY}!important;
    border-bottom:3px solid {GOLD};
  }}

  /* ---- callouts ---- */
  .qsq-note {{
    border-left:4px solid {BLUE}; background:{PALE_BLUE};
    padding:.75rem 1rem; border-radius:0 9px 9px 0; margin:.6rem 0;
    font-size:.9rem; color:{INK};
  }}
  .qsq-flag {{
    border-left:4px solid {GOLD}; background:{PALE_GOLD};
    padding:.75rem 1rem; border-radius:0 9px 9px 0; margin:.6rem 0;
    font-size:.9rem; color:#5A4708;
  }}
  .qsq-verdict {{
    border:2px solid {GOLD}; background:{PALE_GOLD}; border-radius:11px;
    padding:.9rem 1.1rem; margin:.8rem 0; color:{NAVY}; font-weight:600;
  }}

  /* ---- tables ---- */
  .stDataFrame {{ border:1px solid #DCE9F5; border-radius:9px; }}
  thead tr th {{ background:{PALE_BLUE}!important; color:{NAVY}!important;
                 font-weight:650!important; }}

  .stProgress > div > div > div > div {{
    background:linear-gradient(90deg,{BLUE},{GOLD});
  }}
  hr {{ border-color:#E2ECF6; }}
</style>
"""


def hero(title: str, subtitle: str) -> str:
    return f'<div class="qsq-hero"><h1>{title}</h1><p>{subtitle}</p></div>'


def section(title: str) -> str:
    return f'<div class="qsq-section"><div class="bar"></div><h3>{title}</h3></div>'


def note(text: str) -> str:
    return f'<div class="qsq-note">{text}</div>'


def flag(text: str) -> str:
    return f'<div class="qsq-flag">{text}</div>'


def verdict(text: str) -> str:
    return f'<div class="qsq-verdict">{text}</div>'
