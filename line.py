import streamlit as st
import pandas as pd
import os
import io
import datetime
from io import BytesIO

# --- ReportLab Imports ---
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak, Image as RLImage
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# --- Dependency Check ---
try:
    import qrcode
    from PIL import Image as PILImage
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(page_title="AgiloSmartTag Studio", page_icon="🏷️", layout="wide")

# --- Global Bin Info Map ---
bin_info_map = {}

# --- Style Definitions ---
bold_style_v1 = ParagraphStyle(name='Bold_v1', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16, spaceBefore=5, spaceAfter=2)
bold_style_v2 = ParagraphStyle(name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16, spaceBefore=10, spaceAfter=15)
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16, spaceBefore=2, spaceAfter=2)
location_header_style = ParagraphStyle(name='LocationHeader', fontName='Helvetica', fontSize=16, alignment=TA_CENTER, leading=18)
location_value_style_base = ParagraphStyle(name='LocationValue_Base', fontName='Helvetica', alignment=TA_CENTER)
bin_bold_style = ParagraphStyle(name='Bold', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, leading=14)
bin_desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
bin_qty_style = ParagraphStyle(name='Quantity', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
rl_header_style = ParagraphStyle(name='RL_Header', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)
rl_cell_left_style = ParagraphStyle(name='RL_Cell_Left', fontName='Helvetica', fontSize=9, alignment=TA_LEFT)

# --- Helper Functions ---
def parse_dimensions(dim_str):
    if not dim_str: return (0, 0)
    try:
        dim_str = dim_str.lower().replace('mm', '').strip()
        parts = [int(x.strip()) for x in dim_str.split('x')]
        if len(parts) >= 2: return (parts[0], parts[1])
        return (0, 0)
    except: return (0, 0)

def format_part_no_v1(part_no):
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=17>{part1}</font><font size=22>{part2}</font></b>", bold_style_v1)
    return Paragraph(f"<b><font size=17>{part_no}</font></b>", bold_style_v1)

def format_part_no_v2(part_no):
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if part_no.upper() == 'EMPTY': return Paragraph(f"<b><font size=34>EMPTY</font></b><br/><br/>", bold_style_v2)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=34>{part1}</font><font size=40>{part2}</font></b><br/><br/>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b><br/><br/>", bold_style_v2)

def format_description_v1(desc):
    if not desc or not isinstance(desc, str): desc = str(desc)
    font_size = 15 if len(desc) <= 30 else 13 if len(desc) <= 50 else 11 if len(desc) <= 70 else 10 if len(desc) <= 90 else 9
    desc_style_v1 = ParagraphStyle(name='Description_v1', fontName='Helvetica', fontSize=font_size, alignment=TA_LEFT, leading=font_size + 2)
    return Paragraph(desc, desc_style_v1)

def format_description(desc):
    if not desc or not isinstance(desc, str): desc = str(desc)
    return Paragraph(desc, desc_style)

def get_dynamic_location_style(text, column_type):
    text_len = len(str(text))
    font_size, leading = 16, 18
    if column_type == 'Bus Model':
        if text_len <= 3: font_size, leading = 14, 18
        elif text_len <= 5: font_size, leading = 12, 18
        elif text_len <= 10: font_size, leading = 10, 15
        else: font_size, leading = 11, 13
    elif column_type == 'Station No':
        if text_len <= 2: font_size, leading = 20, 21
        elif text_len <= 5: font_size, leading = 18, 21
        elif text_len <= 8: font_size, leading = 15, 15
        else: font_size, leading = 11, 13
    else:
        if text_len <= 2: font_size, leading = 16, 18
        elif text_len <= 4: font_size, leading = 14, 18
        else: font_size, leading = 12, 14
    return ParagraphStyle(name=f'Dyn_{column_type}_{text_len}', parent=location_value_style_base, fontName='Helvetica', fontSize=font_size, leading=leading, alignment=TA_CENTER)

def find_required_columns(df):
    cols_map = {col.strip().upper(): col for col in df.columns}
    def find_col(patterns):
        for p in patterns:
            if p in cols_map: return cols_map[p]
        return None
    return {
        'Part No': find_col([k for k in cols_map if 'PART' in k and ('NO' in k or 'NUM' in k)]),
        'Description': find_col([k for k in cols_map if 'DESC' in k]),
        'Bus Model': find_col([k for k in cols_map if 'BUS' in k and 'MODEL' in k]),
        'Station No': find_col([k for k in cols_map if 'STATION' in k and 'NAME' not in k]),
        'Station Name': find_col(['STATION NAME', 'ST. NAME', 'STATION_NAME', 'ST_NAME']),
        'Container': find_col([k for k in cols_map if 'CONTAINER' in k]),
        'Qty/Bin': find_col([k for k in cols_map if 'QTY/BIN' in k or 'QTY_BIN' in k or ('QTY' in k and 'BIN' in k)]),
        'Qty/Veh': find_col([k for k in cols_map if 'QTY/VEH' in k or 'QTY_VEH' in k or ('QTY' in k and 'VEH' in k)]),
        'Zone': find_col(['ZONE', 'ABB ZONE', 'ABB_ZONE', 'AREA'])
    }

def get_unique_containers(df, container_col):
    if not container_col or container_col not in df.columns: return []
    return sorted(df[container_col].dropna().astype(str).unique())

def automate_location_assignment(df, base_rack_id, rack_configs, status_text=None):
    required_cols = find_required_columns(df)
    part_no_col, desc_col, model_col, station_col, container_col = required_cols['Part No'], required_cols['Description'], required_cols['Bus Model'], required_cols['Station No'], required_cols['Container']
    if not all([part_no_col, container_col, station_col]):
        st.error("❌ Required columns not found")
        return None
    df_processed = df.copy()
    rename_dict = {part_no_col: 'Part No', desc_col: 'Description', model_col: 'Bus Model', station_col: 'Station No', container_col: 'Container'}
    df_processed.rename(columns={k: v for k, v in rename_dict.items() if k}, inplace=True)
    df_processed['bin_info'] = df_processed['Container'].map(bin_info_map)
    df_processed['bin_area'] = df_processed['bin_info'].apply(lambda x: x['dims'][0] * x['dims'][1] if x and x.get('dims') else 0)
    final_df_parts = []
    available_cells = []
    for rack_name, config in sorted(rack_configs.items()):
        rack_num_val = ''.join(filter(str.isdigit, rack_name))
        rack_num_1st = rack_num_val[0] if len(rack_num_val) > 1 else '0'
        rack_num_2nd = rack_num_val[1] if len(rack_num_val) > 1 else rack_num_val[0]
        rack_dims, cells_per_level = config.get('dims', (0, 0)), config.get('cells_per_level', 0)
        rack_area = rack_dims[0] * rack_dims[1]
        cell_area = rack_area / cells_per_level if rack_area and cells_per_level else 0
        for level in sorted(config.get('levels', [])):
            for i in range(cells_per_level):
                available_cells.append({'Rack': base_rack_id, 'Rack No 1st': rack_num_1st, 'Rack No 2nd': rack_num_2nd, 'Level': level, 'Physical_Cell': f"{i + 1:02d}", 'cell_area': cell_area})
    current_cell_index, last_processed_station = 0, "N/A"
    for station_no, station_group in df_processed.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Processing station: {station_no}...")
        last_processed_station = station_no
        sorted_groups = sorted(station_group.groupby('Container'), key=lambda x: x[1]['bin_area'].iloc[0], reverse=True)
        for container_type, group_df in sorted_groups:
            parts = group_df.to_dict('records')
            bin_dims = bin_info_map.get(container_type, {}).get('dims', (0, 0))
            bin_area = bin_dims[0] * bin_dims[1]
            if bin_area == 0: continue
            while parts:
                if current_cell_index >= len(available_cells): break
                cell = available_cells[current_cell_index]
                max_bins = max(1, int(cell['cell_area'] // bin_area) if cell['cell_area'] else 1)
                chunk = parts[:max_bins]
                parts = parts[max_bins:]
                for part in chunk:
                    part.update(cell)
                    final_df_parts.append(part)
                current_cell_index += 1
        if current_cell_index >= len(available_cells): break
    for i in range(current_cell_index, len(available_cells)):
        empty_part = {'Part No': 'EMPTY', 'Description': '', 'Bus Model': '', 'Station No': last_processed_station, 'Container': ''}
        empty_part.update(available_cells[i])
        final_df_parts.append(empty_part)
    return pd.DataFrame(final_df_parts)

def create_location_key(row):
    return '_'.join([str(row.get(c, '')) for c in ['Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']])

def extract_location_values(row):
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']]

# PDF generation functions would continue here (abbreviated for length)
# Including: generate_rack_labels_v1, generate_rack_labels_v2, generate_bin_labels, generate_rack_list_pdf

# --- Main Application ---
def main():
    st.title("🏷️ AgiloSmartTag Studio - Line Automation System")
    st.markdown("<p style='font-style:italic;'>Designed and Developed by Agilomatrix</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.sidebar.title("🏭 Line Configuration")
    output_type = st.sidebar.selectbox("Output Type:", ["Rack Labels", "Bin Labels", "Rack List"])
    rack_label_format = "Single Part"
    if output_type == "Rack Labels":
        rack_label_format = st.sidebar.selectbox("Label Format:", ["Single Part", "Multiple Parts"])
    
    base_rack_id = st.sidebar.text_input("Infrastructure ID", "R", help="Example: R=Rack, TR=Tray, SH=Shelving")
    
    uploaded_file = st.file_uploader("📂 Upload Excel/CSV File", type=['xlsx', 'xls', 'csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, dtype=str) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file, dtype=str)
            df.fillna('', inplace=True)
            st.success(f"✅ File loaded! Found {len(df)} rows")
            
            required_cols = find_required_columns(df)
            if required_cols['Container']:
                unique_containers = get_unique_containers(df, required_cols['Container'])
                
                with st.expander("⚙️ Configure Line Infrastructure", expanded=True):
                    st.subheader("📦 Container Dimensions")
                    bin_dims = {}
                    cols = st.columns(min(3, len(unique_containers)))
                    for idx, container in enumerate(unique_containers):
                        with cols[idx % 3]:
                            dim = st.text_input(f"{container}", key=f"dim_{container}", placeholder="300x200")
                            bin_dims[container] = dim
                    
                    st.markdown("---")
                    st.subheader("🗄️ Rack Configuration")
                    num_racks = st.number_input("Number of Racks per Station", 1, 20, 2)
                    
                    rack_configs = {}
                    for i in range(num_racks):
                        with st.expander(f"Rack {i+1:02d} Configuration"):
                            col1, col2 = st.columns(2)
                            with col1:
                                rack_dim = st.text_input(f"Dimensions", key=f"rack_{i}", placeholder="1200x1000")
                                levels = st.multiselect(f"Levels", ['A','B','C','D','E','F','G','H'], default=['A','B','C','D','E'], key=f"lvl_{i}")
                            with col2:
                                cells_per_level = st.number_input(f"Cells per Level", 1, 20, 4, key=f"cells_{i}")
                            
                            rack_name = f"Rack {i+1:02d}"
                            if rack_dim and levels:
                                dims = parse_dimensions(rack_dim)
                                rack_configs[rack_name] = {'dims': dims, 'levels': levels, 'cells_per_level': cells_per_level}
                    
                    # Store bin dimensions
                    for container, dim in bin_dims.items():
                        if dim:
                            bin_info_map[container] = {'dims': parse_dimensions(dim)}
                
                if st.button("🚀 Generate PDF", type="primary"):
                    if not rack_configs or not bin_info_map:
                        st.error("❌ Please configure all dimensions")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        try:
                            df_processed = automate_location_assignment(df, base_rack_id, rack_configs, status_text)
                            if df_processed is not None and not df_processed.empty:
                                status_text.text("✅ Processing complete!")
                                st.success(f"✅ Generated {len(df_processed)} location assignments")
                                st.dataframe(df_processed.head(20))
                            else:
                                st.error("❌ No data processed")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                            st.exception(e)
                        finally:
                            progress_bar.empty()
                            status_text.empty()
            else:
                st.error("❌ Container column required")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
    else:
        st.info("👆 Upload a file to begin")

if __name__ == "__main__":
    main()
