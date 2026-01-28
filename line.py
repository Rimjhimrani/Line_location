import streamlit as st
import pandas as pd
import os
import io
import re
import math
import datetime
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak, Image as RLImage
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# QR Code imports
try:
    import qrcode
    from PIL import Image
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(
    page_title="AgiloSmartTag Studio",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Simple, User-Friendly UI ---
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Main title */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(120deg, #2E86AB, #A23B72, #F18F01);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
        animation: fadeIn 1s ease-in;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #555;
        margin-bottom: 2rem;
    }
    
    /* Step cards */
    .step-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 2rem 0;
        border-left: 8px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .step-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    
    .step-number {
        display: inline-block;
        width: 50px;
        height: 50px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 50px;
        font-size: 1.5rem;
        font-weight: bold;
        margin-right: 1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .step-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2E86AB;
        margin: 1rem 0;
        display: flex;
        align-items: center;
    }
    
    .step-description {
        font-size: 1.1rem;
        color: #666;
        line-height: 1.6;
        margin: 1rem 0;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        font-size: 1.1rem;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
    }
    
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        font-size: 1.2rem;
        text-align: center;
        box-shadow: 0 5px 20px rgba(17, 153, 142, 0.3);
        animation: slideIn 0.5s ease-out;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        font-size: 1.1rem;
        box-shadow: 0 5px 20px rgba(245, 87, 108, 0.3);
    }
    
    .help-box {
        background: #FFF9E6;
        border-left: 5px solid #FFD700;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 1rem;
        color: #666;
    }
    
    /* Big buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1.2rem 3rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.3rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.5);
    }
    
    /* Input fields */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input {
        font-size: 1.1rem;
        padding: 0.8rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus,
    .stNumberInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* File uploader */
    .uploadedFile {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        background: #f8f9ff;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Simple label */
    .simple-label {
        font-size: 1.2rem;
        font-weight: 600;
        color: #333;
        margin: 0.5rem 0;
    }
    
    /* Example text */
    .example-text {
        font-size: 0.95rem;
        color: #888;
        font-style: italic;
        margin-top: 0.3rem;
    }
    
    /* Container cards */
    .container-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .container-card:hover {
        border-color: #667eea;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.1);
    }
    
    /* Metric display */
    .big-metric {
        text-align: center;
        padding: 1.5rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .big-metric-value {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .big-metric-label {
        font-size: 1rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Style Definitions (Same as before) ---
bold_style_v2 = ParagraphStyle(
    name='Bold_v2',
    fontName='Helvetica-Bold',
    fontSize=10,
    alignment=TA_LEFT,
    leading=32, 
    spaceBefore=0,
    spaceAfter=2,
    wordWrap='CJK'
)
desc_style = ParagraphStyle(
    name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16, spaceBefore=2, spaceAfter=2
)
bin_bold_style = ParagraphStyle(
    name='BinBold',
    fontName='Helvetica-Bold',
    fontSize=18,
    alignment=TA_CENTER,
    leading=20
)
bin_desc_style = ParagraphStyle(
    name='BinDesc',
    fontName='Helvetica',
    fontSize=10,
    alignment=TA_CENTER,
    leading=12
)
bin_qty_style = ParagraphStyle(
    name='BinQty',
    fontName='Helvetica-Bold',
    fontSize=14,
    alignment=TA_CENTER,
    leading=16
)
rl_header_style = ParagraphStyle(
    name='RLHeader',
    fontName='Helvetica-Bold',
    fontSize=11,
    alignment=TA_LEFT
)
rl_cell_left_style = ParagraphStyle(
    name='RLCellLeft',
    fontName='Helvetica',
    fontSize=10,
    alignment=TA_LEFT
)

# --- All the same functions from original code ---
def format_part_no_v2(part_no):
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if part_no.upper() == 'EMPTY':
         return Paragraph(f"<b><font size=34>EMPTY</font></b>", bold_style_v2)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=34>{part1}</font><font size=40>{part2}</font></b>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b>", bold_style_v2)

def format_description(desc):
    if not desc or not isinstance(desc, str): desc = str(desc)
    return Paragraph(desc, desc_style)

def find_required_columns(df):
    cols = {col.upper().strip(): col for col in df.columns}
    part_no_key = next((k for k in cols if 'PART' in k and ('NO' in k or 'NUM' in k)), None)
    desc_key = next((k for k in cols if 'DESC' in k), None)
    bus_model_key = next((k for k in cols if 'BUS' in k and 'MODEL' in k), None)
    station_no_key = next((k for k in cols if 'STATION' in k and 'NO' in k), None)
    if not station_no_key:
        station_no_key = next((k for k in cols if 'STATION' in k), None)
    container_type_key = next((k for k in cols if 'CONTAINER' in k), None)
    station_name_key = next((k for k in cols if 'STATION' in k and 'NAME' in k and 'SHORT' not in k), None)
    
    return {
        'Part No': cols.get(part_no_key),
        'Description': cols.get(desc_key),
        'Bus Model': cols.get(bus_model_key),
        'Station No': cols.get(station_no_key),
        'Container': cols.get(container_type_key),
        'Station Name': cols.get(station_name_key)
    }

def get_unique_containers(df, container_col):
    if not container_col or container_col not in df.columns: return []
    return sorted(df[container_col].dropna().astype(str).unique())

def parse_dimensions(dim_str):
    if not isinstance(dim_str, str) or not dim_str: return 0, 0
    nums = [int(n) for n in re.findall(r'\d+', dim_str)]
    return (nums[0], nums[1]) if len(nums) >= 2 else (0, 0)

def calculate_bins_per_level(rack_dims, container_dims):
    rack_l, rack_w = parse_dimensions(rack_dims)
    cont_l, cont_w = parse_dimensions(container_dims)
    
    if rack_l == 0 or rack_w == 0 or cont_l == 0 or cont_w == 0:
        return 0
    
    bins_option1 = (rack_l // cont_l) * (rack_w // cont_w)
    bins_option2 = (rack_l // cont_w) * (rack_w // cont_l)
    
    return max(bins_option1, bins_option2)

def find_rack_type_for_container(container_type, rack_templates, container_configs):
    for rack_name, config in rack_templates.items():
        manual_capacity = config['capacities'].get(container_type, 0)
        
        if manual_capacity > 0:
            return rack_name, config, manual_capacity
    
    return None, None, 0

def generate_by_rack_type(df, base_rack_id, rack_templates, container_configs, status_text=None):
    req = find_required_columns(df)
    df_p = df.copy()
    
    rename_dict = {}
    if req['Part No']: rename_dict[req['Part No']] = 'Part No'
    if req['Description']: rename_dict[req['Description']] = 'Description'
    if req['Bus Model']: rename_dict[req['Bus Model']] = 'Bus Model'
    if req['Station No']: rename_dict[req['Station No']] = 'Station No'
    if req['Container']: rename_dict[req['Container']] = 'Container'
    if req['Station Name']: rename_dict[req['Station Name']] = 'Station Name'
    
    df_p.rename(columns=rename_dict, inplace=True)
    
    final_data = []
    if not rack_templates: return pd.DataFrame()

    for station_no, station_group in df_p.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Processing Station: {station_no}...")
        
        station_rack_num = 1
        rack_type_states = {}
        
        for rack_name in rack_templates.keys():
            rack_type_states[rack_name] = {
                'curr_lvl_idx': 0,
                'curr_cell_idx': 1,
                'curr_rack_num': station_rack_num
            }
        
        for cont_type, parts_group in station_group.groupby('Container', sort=True):
            rack_name, config, bins_per_level = find_rack_type_for_container(
                cont_type, rack_templates, container_configs
            )
            
            if bins_per_level == 0 or rack_name is None:
                if status_text: 
                    status_text.text(f"⚠️ Skipping {cont_type} at ST-{station_no}")
                continue
            
            levels = config['levels']
            state = rack_type_states[rack_name]
            
            all_parts = parts_group.to_dict('records')

            for part in all_parts:
                if state['curr_cell_idx'] > bins_per_level:
                    state['curr_cell_idx'] = 1
                    state['curr_lvl_idx'] += 1
                
                if state['curr_lvl_idx'] >= len(levels):
                    state['curr_lvl_idx'] = 0
                    station_rack_num += 1
                    state['curr_rack_num'] = station_rack_num
                    state['curr_cell_idx'] = 1
                rack_str = f"{state['curr_rack_num']:02d}"
                
                rack_dims = config.get('dims', '')
                display_rack_name = f"{rack_name} ({rack_dims})" if rack_dims else rack_name
                part.update({
                    'Rack': base_rack_id, 
                    'Rack No 1st': rack_str[0], 
                    'Rack No 2nd': rack_str[1],
                    'Level': levels[state['curr_lvl_idx']], 
                    'Physical_Cell': f"{state['curr_cell_idx']:02d}",
                    'Station No': station_no, 
                    'Rack Key': rack_str,
                    'Rack Type': display_rack_name,
                    'Calculated_Capacity': bins_per_level
                })
                final_data.append(part)
                state['curr_cell_idx'] += 1
            
            station_rack_num = state['curr_rack_num']
            if state['curr_cell_idx'] > 1 or state['curr_lvl_idx'] > 0:
                station_rack_num += 1
    
    return pd.DataFrame(final_data)

def generate_allocation_summary(df, rack_templates):
    if df.empty:
        return pd.DataFrame()
    
    summary_data = []
    
    for station_no in sorted(df['Station No'].unique()):
        station_df = df[df['Station No'] == station_no]
        row_data = {'Station Number': f'ST - {station_no}'}
        
        for rack_name, config in rack_templates.items():
            rack_dims = config.get('dims', '')
            display_name = f"{rack_name} ({rack_dims})" if rack_dims else rack_name
            
            rack_type_df = station_df[station_df['Rack Type'] == display_name] if 'Rack Type' in station_df.columns else pd.DataFrame()
            
            col_name = display_name
            if not rack_type_df.empty:
                rack_count = rack_type_df['Rack Key'].nunique()
                row_data[col_name] = rack_count if rack_count > 0 else ''
            else:
                row_data[col_name] = ''
        
        summary_data.append(row_data)
    
    summary_df = pd.DataFrame(summary_data)
    
    if not summary_df.empty:
        total_row = {'Station Number': 'TOTAL'}
        for col in summary_df.columns:
            if col != 'Station Number':
                numeric_values = pd.to_numeric(summary_df[col], errors='coerce').fillna(0)
                total = int(numeric_values.sum())
                total_row[col] = total if total > 0 else ''
        summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)
    
    return summary_df

def assign_sequential_location_ids(df):
    if df.empty: return df
    df_sorted = df.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']).copy()
    df_parts_only = df_sorted[df_sorted['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    
    location_counters = {}
    sequential_ids = []
    
    for _, row in df_parts_only.iterrows():
        key = (row['Station No'], row['Rack No 1st'], row['Rack No 2nd'], row['Level'])
        if key not in location_counters: location_counters[key] = 1
        sequential_ids.append(location_counters[key])
        location_counters[key] += 1
        
    df_parts_only['Cell'] = sequential_ids
    df_empty_only = df_sorted[df_sorted['Part No'].astype(str).str.upper() == 'EMPTY'].copy()
    df_empty_only['Cell'] = df_empty_only['Physical_Cell']
    
    return pd.concat([df_parts_only, df_empty_only], ignore_index=True)

def extract_location_values(row):
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']]

def extract_store_location_data_from_excel(row):
    def get_clean_value(possible_names, default=''):
        for name in possible_names:
            val = row.get(name)
            if pd.notna(val) and str(val).strip().lower() not in ['nan', 'none', 'null', '']:
                return str(val).strip()
        return default

    store_location = get_clean_value(['Store Location', 'STORELOCATION', 'Store_Location'])
    zone = get_clean_value(['ABB ZONE', 'ABB_ZONE', 'ABBZONE', 'Zone'])
    location = get_clean_value(['ABB LOCATION', 'ABB_LOCATION', 'ABBLOCATION'])
    floor = get_clean_value(['ABB FLOOR', 'ABB_FLOOR', 'ABBFLOOR'])
    rack_no = get_clean_value(['ABB RACK NO', 'ABB_RACK_NO', 'ABBRACKNO'])
    level_in_rack = get_clean_value(['ABB LEVEL IN RACK', 'ABB_LEVEL_IN_RACK', 'ABBLEVELINRACK'])
    station_name_short = get_clean_value(['ST. NAME (Short)', 'ST.NAME (Short)', 'ST NAME (Short)', 'Station Name Short']) 
    
    return [store_location, station_name_short, zone, location, floor, rack_no, level_in_rack]

def generate_qr_code_image(data):
    if not QR_AVAILABLE:
        return None
    try:
        qr = qrcode.QRCode(version=1, box_size=10, border=2)
        qr.add_data(data)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        return RLImage(img_buffer, width=2.5*cm, height=2.5*cm)
    except:
        return None

def generate_rack_labels(df, progress_bar=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    
    df.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    df_parts_only = df[df['Part No'].astype(str).str.upper() != 'EMPTY'].copy()

    total_labels = len(df_parts_only)
    label_count = 0

    for i, part in enumerate(df_parts_only.to_dict('records')):
        if progress_bar: progress_bar.progress(int((i / total_labels) * 100))
        
        if label_count > 0 and label_count % 4 == 0: elements.append(PageBreak())

        part_table = Table([['Part No', format_part_no_v2(str(part['Part No']))], ['Description', format_description(str(part['Description']))]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        location_values = extract_location_values(part)
        location_data = [['Line Location'] + location_values]
        col_widths = [1.7, 2.9, 1.3, 1.2, 1.3, 1.3, 1.3]
        location_widths = [4 * cm] + [w * (11 * cm) / sum(col_widths) for w in col_widths]
        location_table = Table(location_data, colWidths=location_widths, rowHeights=1.2*cm)
        
        part_table.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (0, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('FONTSIZE', (0, 0), (0, -1), 16)]))
        location_colors = [colors.HexColor('#E9967A'), colors.HexColor('#ADD8E6'), colors.HexColor('#90EE90'), colors.HexColor('#FFD700'), colors.HexColor('#ADD8E6'), colors.HexColor('#E9967A'), colors.HexColor('#90EE90')]
        location_style = [('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('FONTSIZE', (0, 0), (-1, -1), 16)]
        for j, color in enumerate(location_colors): location_style.append(('BACKGROUND', (j+1, 0), (j+1, 0), color))
        location_table.setStyle(TableStyle(location_style))
        
        elements.append(part_table); elements.append(Spacer(1, 0.3 * cm)); elements.append(location_table); elements.append(Spacer(1, 0.2 * cm))
        label_count += 1
        
    if elements: doc.build(elements)
    buffer.seek(0)
    summary = df_parts_only.groupby(['Station No', 'Rack No 1st', 'Rack No 2nd']).size().reset_index(name='Labels')
    summary['Rack'] = summary['Rack No 1st'] + summary['Rack No 2nd']
    return buffer, summary[['Station No', 'Rack', 'Labels']]

def generate_bin_labels(df, mtm_models, progress_bar=None, status_text=None):
    if not QR_AVAILABLE:
        st.error("❌ QR Code library not found. Please install `qrcode` and `Pillow`.")
        return None, {}

    STICKER_WIDTH, STICKER_HEIGHT = 10 * cm, 15 * cm
    CONTENT_BOX_WIDTH, CONTENT_BOX_HEIGHT = 10 * cm, 7.2 * cm
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(STICKER_WIDTH, STICKER_HEIGHT),
                            topMargin=0.2*cm, bottomMargin=STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm,
                            leftMargin=0.1*cm, rightMargin=0.1*cm)

    df_filtered = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    df_filtered.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    total_labels = len(df_filtered)
    label_summary = {}
    all_elements = []

    def draw_border(canvas, doc):
        canvas.saveState()
        x_offset = (STICKER_WIDTH - CONTENT_BOX_WIDTH) / 2
        y_offset = STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm
        canvas.setStrokeColorRGB(0, 0, 0)
        canvas.setLineWidth(1.8)
        canvas.rect(x_offset + doc.leftMargin, y_offset, CONTENT_BOX_WIDTH - 0.2*cm, CONTENT_BOX_HEIGHT)
        canvas.restoreState()

    for i, row in enumerate(df_filtered.to_dict('records')):
        if progress_bar: progress_bar.progress(int(((i+1) / total_labels) * 100))
        if status_text: status_text.text(f"Creating Label {i+1} of {total_labels}")
        
        rack_key = f"ST-{row.get('Station No', 'NA')} / Rack {row.get('Rack No 1st', '0')}{row.get('Rack No 2nd', '0')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1

        part_no = str(row.get('Part No', ''))
        desc = str(row.get('Description', ''))
        qty_bin = str(row.get('Qty/Bin', ''))
        qty_veh = str(row.get('Qty/Veh', ''))

        store_loc_raw = extract_store_location_data_from_excel(row)
        line_loc_raw = extract_location_values(row)

        store_loc_str = "|".join([str(x).strip() for x in store_loc_raw])
        line_loc_str = "|".join([str(x).strip() for x in line_loc_raw])

        qr_data = (
            f"Part No: {part_no}\n"
            f"Desc: {desc}\n"
            f"Qty/Bin: {qty_bin}\n"
            f"Qty/Veh: {qty_veh}\n"
            f"Store Loc: {store_loc_str}\n"
            f"Line Loc: {line_loc_str}"
        )
        qr_image = generate_qr_code_image(qr_data)
        
        content_width = CONTENT_BOX_WIDTH - 0.2*cm
        
        main_table = Table([
            ["Part No", Paragraph(f"{part_no}", bin_bold_style)],
            ["Description", Paragraph(desc[:47] + "..." if len(desc) > 50 else desc, bin_desc_style)],
            ["Qty/Bin", Paragraph(qty_bin, bin_qty_style)]
        ], colWidths=[content_width/3, content_width*2/3], rowHeights=[0.9*cm, 1.0*cm, 0.5*cm])
        main_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black),('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTNAME', (0,0),(0,-1), 'Helvetica'), ('FONTSIZE', (0,0),(0,-1), 11)]))

        inner_table_width = content_width * 2 / 3
        col_props = [1.8, 2.4, 0.7, 0.7, 0.7, 0.7, 0.9]
        inner_col_widths = [w * inner_table_width / sum(col_props) for w in col_props]
        
        store_loc_values = extract_store_location_data_from_excel(row)
        store_loc_inner = Table([store_loc_values], colWidths=inner_col_widths, rowHeights=[0.5*cm])
        store_loc_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTNAME', (0,0),(-1,-1), 'Helvetica-Bold'), ('FONTSIZE', (0,0),(-1,-1), 9)]))
        store_loc_table = Table([[Paragraph("Store Location", bin_desc_style), store_loc_inner]], colWidths=[content_width/3, inner_table_width], rowHeights=[0.5*cm])
        store_loc_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        
        line_loc_values = extract_location_values(row)
        line_loc_inner = Table([line_loc_values], colWidths=inner_col_widths, rowHeights=[0.5*cm])
        line_loc_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTNAME', (0,0),(-1,-1), 'Helvetica-Bold'), ('FONTSIZE', (0,0),(-1,-1), 9)]))
        line_loc_table = Table([[Paragraph("Line Location", bin_desc_style), line_loc_inner]], colWidths=[content_width/3, inner_table_width], rowHeights=[0.5*cm])
        line_loc_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))

        mtm_table = None
        if mtm_models:
            qty_veh = str(row.get('Qty/Veh', ''))
            bus_model = str(row.get('Bus Model', '')).strip().upper()
            
            mtm_qty_values = []
            for model in mtm_models:
                if bus_model == model.strip().upper() and qty_veh:
                    mtm_qty_values.append(Paragraph(f"<b>{qty_veh}</b>", bin_qty_style))
                else:
                    mtm_qty_values.append("")
            
            mtm_data = [mtm_models, mtm_qty_values]
            num_models = len(mtm_models)
            total_mtm_width = 3.6 * cm
            col_width = total_mtm_width / num_models if num_models > 0 else total_mtm_width

            mtm_table = Table(mtm_data, colWidths=[col_width] * num_models, rowHeights=[0.75*cm, 0.75*cm])
            mtm_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTNAME', (0,0),(-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0),(-1,-1), 9)]))

        mtm_width, qr_width, gap_width = 3.6 * cm, 2.5 * cm, 1.0 * cm
        remaining_width = content_width - mtm_width - gap_width - qr_width
        
        bottom_row_content = [mtm_table if mtm_table else "", "", qr_image or "", ""]

        bottom_row = Table(
            [bottom_row_content],
            colWidths=[mtm_width, gap_width, qr_width, remaining_width],
            rowHeights=[2.5*cm]
        )
        bottom_row.setStyle(TableStyle([('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))

        all_elements.extend([main_table, store_loc_table, line_loc_table, Spacer(1, 0.2*cm), bottom_row])
        if i < total_labels - 1:
            all_elements.append(PageBreak())

    if all_elements: doc.build(all_elements, onFirstPage=draw_border, onLaterPages=draw_border)
    buffer.seek(0)
    return buffer, label_summary

def generate_rack_list_pdf(df, base_rack_id, top_logo_file, top_logo_w, top_logo_h, fixed_logo_path, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=0.5*cm, bottomMargin=0.5*cm, leftMargin=1*cm, rightMargin=1*cm)
    elements = []
    
    df = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    df['Rack Key'] = df.apply(lambda x: f"{x.get('Rack No 1st', '')}{x.get('Rack No 2nd', '')}", axis=1)
    df.sort_values(by=['Station No', 'Rack Key', 'Level', 'Cell'], inplace=True)
    grouped = df.groupby(['Station No', 'Rack Key'])
    total_groups = len(grouped)
    has_zone = 'Zone' in df.columns and df['Zone'].notna().any()
    
    master_value_style_left = ParagraphStyle(name='MasterValLeft', fontName='Helvetica-Bold', fontSize=13, alignment=TA_LEFT)
    master_value_style_center = ParagraphStyle(name='MasterValCenter', fontName='Helvetica-Bold', fontSize=13, alignment=TA_CENTER)

    for i, ((station_no, rack_key), group) in enumerate(grouped):
        if progress_bar: progress_bar.progress(int(((i+1) / total_groups) * 100))
        if status_text: status_text.text(f"Creating List for Station {station_no}, Rack {rack_key}")
        
        first_row = group.iloc[0]
        station_name = str(first_row.get('Station Name', ''))
        bus_model = str(first_row.get('Bus Model', ''))
        
        top_logo_img = ""
        if top_logo_file:
            try:
                img_io = io.BytesIO(top_logo_file.getvalue())
                top_logo_img = RLImage(img_io, width=top_logo_w*cm, height=top_logo_h*cm)
            except:
                pass
        
        header_table = Table([[Paragraph("Document Ref No.:", rl_header_style), "", top_logo_img]], colWidths=[5*cm, 17.5*cm, 5*cm])
        header_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('ALIGN', (-1,-1), (-1,-1), 'RIGHT'),
        ]))
        elements.append(header_table)
        elements.append(Spacer(1, 0.1*cm))
        
        master_data = [
            [Paragraph("STATION NAME", ParagraphStyle('H', fontName='Helvetica-Bold', fontSize=12)), 
             Paragraph(station_name, master_value_style_left),
             Paragraph("STATION NO", ParagraphStyle('H', fontName='Helvetica-Bold', fontSize=13)), 
             Paragraph(str(station_no), master_value_style_center)],
            
            [Paragraph("MODEL", ParagraphStyle('H', fontName='Helvetica-Bold', fontSize=13)), 
             Paragraph(bus_model, master_value_style_left),
             Paragraph("RACK NO", ParagraphStyle('H', fontName='Helvetica-Bold', fontSize=13)), 
             Paragraph(f"Rack - {rack_key}", master_value_style_center)]
        ]
        
        bg_blue = colors.HexColor("#8EAADB")
        
        master_table = Table(master_data, colWidths=[4*cm, 9.5*cm, 4*cm, 10*cm], rowHeights=[0.8*cm, 0.8*cm])
        master_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('BACKGROUND', (0,0), (-1,-1), bg_blue),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
            ('ALIGN', (0,0), (0,-1), 'LEFT'), 
            ('ALIGN', (2,0), (2,-1), 'LEFT'),
            ('ALIGN', (1,0), (1,-1), 'LEFT'), 
            ('ALIGN', (3,0), (3,-1), 'CENTER'),
        ]))
        elements.append(master_table)
        
        header_row = ["S.NO", "PART NO", "PART DESCRIPTION", "CONTAINER", "QTY/BIN", "LOCATION"]
        col_widths = [1.5*cm, 4.5*cm, 9.5*cm, 3.5*cm, 2.5*cm, 6.0*cm]
        
        if has_zone:
            header_row.insert(0, "ZONE")
            col_widths = [2.0*cm, 1.3*cm, 4.0*cm, 8.2*cm, 3.5*cm, 2.5*cm, 6.0*cm]
            
        data_rows = [header_row]
        group_sorted = group.sort_values(by=['Level', 'Cell'])
        bg_orange = colors.HexColor("#F4B084") 
        
        table_style_cmds = [
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('BACKGROUND', (0,0), (-1,0), bg_orange), 
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTSIZE', (0,0), (-1,-1), 11),
            ('TOPPADDING', (0,0), (-1,-1), 1),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ]
        
        previous_zone = None
        current_zone_start = 1
        
        for idx, row in enumerate(group_sorted.to_dict('records')):
            s_no = idx + 1
            p_no = str(row.get('Part No', ''))
            desc = str(row.get('Description', ''))
            cont = str(row.get('Container', ''))
            qty = str(row.get('Qty/Bin', ''))
            loc_str = f"{bus_model}-{row.get('Station No','')}-{base_rack_id}{rack_key}-{row.get('Level','')}{row.get('Cell','')}"
            
            row_data = [str(s_no), p_no, Paragraph(desc, rl_cell_left_style), cont, qty, loc_str]
            
            if has_zone:
                zone_val = str(row.get('Zone', ''))
                row_data.insert(0, zone_val)
                if zone_val == previous_zone and idx > 0:
                    row_data[0] = "" 
                else:
                    if idx > 0:
                        if current_zone_start < idx + 1: 
                            table_style_cmds.append(('SPAN', (0, current_zone_start), (0, idx)))
                    current_zone_start = idx + 1
                    previous_zone = zone_val
            
            data_rows.append(row_data)
            
        if has_zone and current_zone_start < len(data_rows):
            table_style_cmds.append(('SPAN', (0, current_zone_start), (0, len(data_rows)-1)))

        data_table = Table(data_rows, colWidths=col_widths)
        data_table.setStyle(TableStyle(table_style_cmds))
        elements.append(data_table)
        elements.append(Spacer(1, 0.2*cm))
        
        today_date = datetime.date.today().strftime("%d-%m-%Y")
        
        fixed_logo_img = Paragraph("<b>[Agilomatrix Logo Missing]</b>", rl_cell_left_style)
        if os.path.exists(fixed_logo_path):
            try:
                 fixed_logo_img = RLImage(fixed_logo_path, width=4.3*cm, height=1.5*cm)
            except:
                 pass
        
        left_content = [
            Paragraph(f"<i>Creation Date: {today_date}</i>", rl_cell_left_style),
            Spacer(1, 0.2*cm),
            Paragraph("<b>Verified by:</b>", ParagraphStyle('BoldFooter', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)),
            Paragraph("Name: ___________________", rl_cell_left_style),
            Paragraph("Signature: _______________", rl_cell_left_style)
        ]

        designed_by_text = Paragraph("Designed by:", ParagraphStyle('DesignedBy', fontName='Helvetica', fontSize=10, alignment=TA_LEFT))
        right_inner_table = Table([[designed_by_text, fixed_logo_img]], colWidths=[3*cm, 4.5*cm])
        right_inner_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (0,0), (0,0), 'RIGHT'),
            ('ALIGN', (1,0), (1,0), 'RIGHT'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))

        footer_table = Table([[left_content, right_inner_table]], colWidths=[20*cm, 7.7*cm])
        footer_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        
        elements.append(footer_table)
        elements.append(PageBreak())
        
    doc.build(elements)
    buffer.seek(0)
    return buffer, total_groups

# --- SIMPLIFIED MAIN UI FOR NON-TECHNICAL USERS ---
def main():
    # Title with icon
    st.markdown('<h1 class="main-title">🏷️ Label Maker</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Create Professional Rack & Bin Labels in 3 Easy Steps</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    
    # STEP 1: Upload File
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title"><span class="step-number">1</span> Upload Your Excel File</div>', unsafe_allow_html=True)
    st.markdown('<p class="step-description">Select the Excel file that contains your parts data. Make sure it has columns for Station, Container, Part Number, and Description.</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Click to browse or drag and drop your file here",
        type=['xlsx', 'xls', 'csv'],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state['df'] = df
            st.session_state.step = 2
            
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ File Loaded Successfully!</strong><br>
                Found <strong>{len(df)}</strong> parts in your file
            </div>
            """, unsafe_allow_html=True)
            
            # Quick preview
            with st.expander("👀 Click here to preview your data"):
                st.dataframe(df.head(10), use_container_width=True)
                
        except Exception as e:
            st.markdown(f"""
            <div class="warning-box">
                <strong>⚠️ Error Loading File</strong><br>
                {str(e)}<br>
                Please make sure your file is a valid Excel or CSV file.
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Only show next steps if file is uploaded
    if 'df' in st.session_state and st.session_state.step >= 2:
        df = st.session_state['df']
        req_cols = find_required_columns(df)
        
        # Check if required columns exist
        if not req_cols['Container'] or not req_cols['Station No']:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Missing Required Columns</strong><br>
                Your Excel file must have columns for:<br>
                • Station (or Station No)<br>
                • Container (or Container Type)<br>
                <br>
                Please update your file and upload again.
            </div>
            """, unsafe_allow_html=True)
            return
        
        # STEP 2: Choose Label Type
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title"><span class="step-number">2</span> Choose What You Want to Create</div>', unsafe_allow_html=True)
        st.markdown('<p class="step-description">Select the type of labels or list you need to generate.</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 Rack Labels\n\nFor labeling rack shelves", use_container_width=True, key="rack_labels_btn"):
                st.session_state.output_type = "Rack Labels"
                st.session_state.step = 3
                st.rerun()
        
        with col2:
            if st.button("🏷️ Bin Labels\n\nFor labeling containers/bins", use_container_width=True, key="bin_labels_btn"):
                st.session_state.output_type = "Bin Labels"
                st.session_state.step = 3
                st.rerun()
        
        with col3:
            if st.button("📄 Rack List\n\nDetailed inventory list", use_container_width=True, key="rack_list_btn"):
                st.session_state.output_type = "Rack List"
                st.session_state.step = 3
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # STEP 3: Configure and Generate
        if st.session_state.step >= 3 and 'output_type' in st.session_state:
            output_type = st.session_state.output_type
            
            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="step-title"><span class="step-number">3</span> Configure Settings for {output_type}</div>', unsafe_allow_html=True)
            st.markdown('<p class="step-description">Fill in the details below. Don\'t worry, we\'ve set default values that work for most cases!</p>', unsafe_allow_html=True)
            
            # Basic Settings
            st.markdown("### 🏭 Basic Information")
            
            col1, col2 = st.columns(2)
            with col1:
                base_rack_id = st.text_input(
                    "Rack ID Prefix", 
                    "R",
                    help="This letter/code will appear at the start of all rack IDs (default: R)"
                )
            
            with col2:
                if output_type == "Bin Labels":
                    st.markdown('<p class="simple-label">Bus Models (for MTM)</p>', unsafe_allow_html=True)
                    mcol1, mcol2, mcol3 = st.columns(3)
                    model1 = mcol1.text_input("Model 1", "7M", label_visibility="collapsed", key="m1")
                    model2 = mcol2.text_input("Model 2", "9M", label_visibility="collapsed", key="m2")
                    model3 = mcol3.text_input("Model 3", "12M", label_visibility="collapsed", key="m3")
                else:
                    model1, model2, model3 = "7M", "9M", "12M"
            
            st.markdown("---")
            
            # Rack Configuration
            st.markdown("### 🏗️ Rack Setup")
            st.markdown('<div class="help-box">💡 <strong>Tip:</strong> Start with 1 rack type. You can add more if you have different sized racks.</div>', unsafe_allow_html=True)
            
            num_rack_types = st.number_input(
                "How many different rack types do you have?", 
                min_value=1, 
                max_value=5, 
                value=1,
                help="If all your racks are the same size, keep this at 1"
            )
            
            unique_c = get_unique_containers(df, req_cols['Container'])
            rack_templates = {}
            container_configs = {}
            
            for i in range(num_rack_types):
                st.markdown(f'<div class="container-card">', unsafe_allow_html=True)
                st.markdown(f"#### Rack Type {i+1}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    r_name = st.text_input(
                        "Rack Name", 
                        f"Type {chr(65+i)}", 
                        key=f"rn_{i}",
                        help="Give this rack type a name (e.g., Type A, Small Rack, etc.)"
                    )
                    st.markdown('<p class="example-text">Example: Type A, Large Rack, Small Shelf</p>', unsafe_allow_html=True)
                
                with col2:
                    r_dim = st.text_input(
                        "Rack Size (Length x Width in mm)", 
                        "2400x800", 
                        key=f"rd_{i}",
                        help="Enter the size of your rack in millimeters"
                    )
                    st.markdown('<p class="example-text">Example: 2400x800, 3000x1000</p>', unsafe_allow_html=True)
                
                r_levels = st.multiselect(
                    "Which shelves does this rack have?", 
                    ['A','B','C','D','E','F'], 
                    default=['A','B','C','D'], 
                    key=f"rl_{i}",
                    help="Select all the shelf levels in your rack (A is bottom, F is top)"
                )
                
                st.markdown("**How many containers fit on each shelf?**")
                st.markdown('<p class="example-text">Enter the number of containers that can fit side by side on one shelf. Enter 0 if this rack doesn\'t hold that container type.</p>', unsafe_allow_html=True)
                
                caps = {}
                
                # Show containers in a nice grid
                num_containers = len(unique_c)
                cols_per_row = 3
                
                for idx in range(0, num_containers, cols_per_row):
                    cols = st.columns(cols_per_row)
                    for col_idx, c in enumerate(unique_c[idx:idx+cols_per_row]):
                        with cols[col_idx]:
                            manual_cap = st.number_input(
                                f"{c}", 
                                min_value=0,
                                max_value=50,
                                value=4 if idx == 0 else 0,  # Default first container to 4
                                key=f"cap_{i}_{c}",
                                help=f"How many {c} fit per shelf? (0 = doesn't fit)"
                            )
                            caps[c] = manual_cap
                
                rack_templates[r_name] = {
                    'levels': r_levels, 
                    'capacities': caps, 
                    'dims': r_dim
                }
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Container dimensions
            st.markdown("---")
            st.markdown("### 📦 Container Sizes")
            st.markdown('<div class="help-box">💡 <strong>Tip:</strong> Enter the size of your containers in millimeters (Length x Width).</div>', unsafe_allow_html=True)
            
            cols_per_row = 2
            for idx in range(0, len(unique_c), cols_per_row):
                cols = st.columns(cols_per_row)
                for col_idx, c in enumerate(unique_c[idx:idx+cols_per_row]):
                    with cols[col_idx]:
                        container_configs[c] = {
                            'dims': st.text_input(
                                f"{c} Size (L x W)", 
                                "600x400", 
                                key=f"cdim_{c}",
                                help=f"Size of {c} container in millimeters"
                            )
                        }
            
            # Logo upload for Rack List
            if output_type == "Rack List":
                st.markdown("---")
                st.markdown("### 📷 Company Logo (Optional)")
                st.markdown('<p class="step-description">Upload your company logo to appear on the rack list. This is optional.</p>', unsafe_allow_html=True)
                
                top_logo_file = st.file_uploader(
                    "Upload Logo (PNG, JPG)", 
                    type=['png', 'jpg', 'jpeg'],
                    label_visibility="collapsed"
                )
                
                if top_logo_file:
                    col1, col2, col3 = st.columns([1,1,1])
                    with col2:
                        st.image(top_logo_file, width=200)
                    
                    col1, col2 = st.columns(2)
                    top_logo_w = col1.number_input("Logo Width (cm)", 1.0, 8.0, 4.0, 0.5)
                    top_logo_h = col2.number_input("Logo Height (cm)", 0.5, 4.0, 1.5, 0.1)
                else:
                    top_logo_w, top_logo_h = 4.0, 1.5
            else:
                top_logo_file = None
                top_logo_w, top_logo_h = 4.0, 1.5
            
            st.markdown("---")
            
            # Big Generate Button
            st.markdown('<div style="text-align: center; margin: 3rem 0;">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(f"🚀 GENERATE {output_type.upper()}", type="primary", use_container_width=True):
                    
                    # Show processing message
                    st.markdown("""
                    <div class="info-box">
                        <strong>⚙️ Processing...</strong><br>
                        Please wait while we create your labels. This may take a minute depending on the size of your data.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    status = st.empty()
                    
                    # Generate allocation
                    with st.spinner("Processing your data..."):
                        df_a = generate_by_rack_type(df, base_rack_id, rack_templates, container_configs, status)
                        df_final = assign_sequential_location_ids(df_a)
                    
                    if not df_final.empty:
                        st.markdown("""
                        <div class="success-box">
                            <strong style="font-size: 1.5rem;">✅ SUCCESS!</strong><br>
                            Your labels are ready to download
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show summary
                        st.markdown("### 📊 Summary")
                        
                        summary_df = generate_allocation_summary(df_final, rack_templates)
                        if not summary_df.empty:
                            # Show metrics
                            total_row = summary_df[summary_df['Station Number'] == 'TOTAL']
                            if not total_row.empty:
                                cols = st.columns(len(rack_templates) + 1)
                                
                                # Total parts
                                with cols[0]:
                                    st.markdown(f"""
                                    <div class="big-metric">
                                        <div class="big-metric-value">{len(df_final)}</div>
                                        <div class="big-metric-label">Total Parts</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Racks by type
                                for idx, (rack_name, config) in enumerate(rack_templates.items()):
                                    rack_dims = config.get('dims', '')
                                    display_name = f"{rack_name} ({rack_dims})" if rack_dims else rack_name
                                    with cols[idx + 1]:
                                        total_racks = total_row[display_name].values[0] if display_name in total_row.columns else 0
                                        st.markdown(f"""
                                        <div class="big-metric">
                                            <div class="big-metric-value">{total_racks}</div>
                                            <div class="big-metric-label">{rack_name} Racks</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                            
                            # Detailed table
                            with st.expander("📋 View Detailed Breakdown"):
                                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                            
                            # Download summary
                            summary_excel_buf = io.BytesIO()
                            summary_df.to_excel(summary_excel_buf, index=False)
                            
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                st.download_button(
                                    label="📥 Download Summary Report",
                                    data=summary_excel_buf.getvalue(),
                                    file_name="Summary_Report.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                        
                        st.markdown("---")
                        
                        # Download allocation data
                        st.markdown("### 📊 Download Complete Data")
                        ex_buf = io.BytesIO()
                        df_final.to_excel(ex_buf, index=False)
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.download_button(
                                label="📥 Download Complete Allocation (Excel)", 
                                data=ex_buf.getvalue(), 
                                file_name="Complete_Allocation.xlsx",
                                use_container_width=True
                            )
                        
                        st.markdown("---")
                        
                        # Generate PDFs
                        st.markdown(f"### 📄 Download {output_type}")
                        
                        prog = st.progress(0)
                        
                        if output_type == "Rack Labels":
                            with st.spinner("Creating your rack labels..."):
                                pdf, _ = generate_rack_labels(df_final, prog)
                            
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                st.download_button(
                                    "📥 DOWNLOAD RACK LABELS PDF", 
                                    pdf, 
                                    "Rack_Labels.pdf",
                                    use_container_width=True
                                )
                            
                        elif output_type == "Bin Labels":
                            mtm_models = [model.strip() for model in [model1, model2, model3] if model.strip()]
                            with st.spinner("Creating your bin labels..."):
                                pdf, label_summary = generate_bin_labels(df_final, mtm_models, prog, status)
                            
                            if pdf:
                                col1, col2, col3 = st.columns([1, 2, 1])
                                with col2:
                                    st.download_button(
                                        "📥 DOWNLOAD BIN LABELS PDF", 
                                        pdf, 
                                        "Bin_Labels.pdf",
                                        use_container_width=True
                                    )
                                
                                # Show label summary
                                if label_summary:
                                    with st.expander("📋 View Labels Count by Rack"):
                                        summary_df = pd.DataFrame(
                                            list(label_summary.items()), 
                                            columns=['Rack Location', 'Number of Labels']
                                        )
                                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
                                        
                                        summary_buf = io.BytesIO()
                                        summary_df.to_excel(summary_buf, index=False)
                                        st.download_button(
                                            "📥 Download Labels Summary", 
                                            summary_buf.getvalue(), 
                                            "Bin_Labels_Summary.xlsx",
                                            use_container_width=True
                                        )
                        
                        elif output_type == "Rack List":
                            with st.spinner("Creating your rack list..."):
                                pdf, _ = generate_rack_list_pdf(
                                    df_final, base_rack_id, top_logo_file, 
                                    top_logo_w, top_logo_h, "Image.png", prog, status
                                )
                            
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                st.download_button(
                                    "📥 DOWNLOAD RACK LIST PDF", 
                                    pdf, 
                                    "Rack_List.pdf",
                                    use_container_width=True
                                )
                        
                        prog.empty()
                        status.empty()
                        
                        # Success animation
                        st.balloons()
                        
                        # Start over button
                        st.markdown("---")
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            if st.button("🔄 Start Over with New File", use_container_width=True):
                                for key in list(st.session_state.keys()):
                                    del st.session_state[key]
                                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
