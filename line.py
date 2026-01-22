import streamlit as st
import pandas as pd
import os
import io
import re
import datetime
from io import BytesIO

# --- ReportLab Imports ---
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak, Image as RLImage
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# --- Dependency Check for Bin Labels (QR Codes) ---
try:
    import qrcode
    from PIL import Image as PILImage
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(page_title="AgiloSmartTag Studio", page_icon="🏷️", layout="wide")

# --- Style Definitions (Automation Ready) ---
bold_style_v1 = ParagraphStyle(name='Bold_v1', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16)
bold_style_v2 = ParagraphStyle(name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16, spaceBefore=10, spaceAfter=15)
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16)
location_header_style = ParagraphStyle(name='LocationHeader', fontName='Helvetica', fontSize=16, alignment=TA_CENTER, leading=18)
location_value_style_base = ParagraphStyle(name='LocationValue_Base', fontName='Helvetica', alignment=TA_CENTER)
bin_bold_style = ParagraphStyle(name='BinBold', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, leading=14)
bin_desc_style = ParagraphStyle(name='BinDesc', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
bin_qty_style = ParagraphStyle(name='BinQty', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
rl_cell_left_style = ParagraphStyle(name='RL_Cell_Left', fontName='Helvetica', fontSize=9, alignment=TA_LEFT)

# --- Formatting Functions ---
def parse_dimensions(dim_str):
    if not dim_str: return (0, 0)
    nums = re.findall(r'\d+', dim_str)
    if len(nums) >= 2: return (int(nums[0]), int(nums[1]))
    return (0, 0)

def format_part_no_v1(part_no):
    part_no = str(part_no)
    if len(part_no) > 5:
        return Paragraph(f"<b><font size=17>{part_no[:-5]}</font><font size=22>{part_no[-5:]}</font></b>", bold_style_v1)
    return Paragraph(f"<b><font size=17>{part_no}</font></b>", bold_style_v1)

def format_part_no_v2(part_no):
    part_no = str(part_no)
    if part_no.upper() == 'EMPTY': return Paragraph(f"<b><font size=34>EMPTY</font></b><br/><br/>", bold_style_v2)
    if len(part_no) > 5:
        return Paragraph(f"<b><font size=34>{part_no[:-5]}</font><font size=40>{part_no[-5:]}</font></b><br/><br/>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b><br/><br/>", bold_style_v2)

def get_dynamic_location_style(text, column_type):
    text_len = len(str(text))
    f_size = 16
    if column_type == 'Bus Model' and text_len > 5: f_size = 10
    elif column_type == 'Station No' and text_len > 4: f_size = 12
    return ParagraphStyle(name=f'D_{column_type}_{text_len}', parent=location_value_style_base, fontSize=f_size, leading=f_size+2)

def find_required_columns(df):
    cols_map = {col.strip().upper(): col for col in df.columns}
    def find_col(patterns):
        for p in patterns:
            for k in cols_map:
                if p in k: return cols_map[k]
        return None
    return {
        'Part No': find_col(['PART NO', 'PART_NO', 'PARTNUM']),
        'Description': find_col(['DESC']),
        'Bus Model': find_col(['BUS', 'MODEL']),
        'Station No': find_col(['STATION NO', 'STATION_NO']),
        'Container': find_col(['CONTAINER']),
        'Qty/Bin': find_col(['QTY/BIN', 'QTY_BIN']),
        'Qty/Veh': find_col(['QTY/VEH', 'QTY_VEH'])
    }

# --- Core Logic: Infrastructure Automation with Capacity ---
def automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status_text=None):
    cols = find_required_columns(df)
    df_p = df.copy()
    rename_map = {v: k for k, v in cols.items() if v}
    df_p.rename(columns=rename_map, inplace=True)

    # 1. Create Physical Slots
    physical_slots = []
    for rack_name, config in sorted(rack_configs.items()):
        r_nums = re.findall(r'\d+', rack_name)
        r_val = r_nums[0].zfill(2) if r_nums else "01"
        for level in sorted(config['levels']):
            for i in range(1, config['cells_per_level'] + 1):
                physical_slots.append({
                    'Rack': base_rack_id, 'Rack No 1st': r_val[0], 'Rack No 2nd': r_val[1],
                    'Level': level, 'Physical_Cell': f"{i:02d}", 'filled_count': 0
                })

    # 2. Assign Parts based on Capacity
    final_parts = []
    slot_idx = 0
    last_station = "N/A"

    for station_no, s_group in df_p.groupby('Station No'):
        if status_text: status_text.text(f"Assigning Parts for Station: {station_no}")
        last_station = station_no
        parts = s_group.to_dict('records')

        for part in parts:
            if slot_idx >= len(physical_slots): break
            
            cont_type = part.get('Container', '')
            max_cap = bin_info_map.get(cont_type, {}).get('capacity', 1)
            
            current_slot = physical_slots[slot_idx]
            
            # Update part with slot data
            part.update({
                'Rack': current_slot['Rack'], 'Rack No 1st': current_slot['Rack No 1st'],
                'Rack No 2nd': current_slot['Rack No 2nd'], 'Level': current_slot['Level'],
                'Physical_Cell': current_slot['Physical_Cell']
            })
            final_parts.append(part)
            
            # Increment slot occupancy
            current_slot['filled_count'] += 1
            if current_slot['filled_count'] >= max_cap:
                slot_idx += 1

    # 3. Fill remaining slots with EMPTY
    while slot_idx < len(physical_slots):
        empty = {'Part No': 'EMPTY', 'Description': '', 'Bus Model': '', 'Station No': last_station}
        empty.update(physical_slots[slot_idx])
        final_parts.append(empty)
        slot_idx += 1

    return pd.DataFrame(final_parts)

# --- PDF Generation (Standard Layouts) ---
def generate_rack_labels_v1(df, progress_bar=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, margin=1*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY']
    for i, (_, row) in enumerate(df_f.iterrows()):
        if progress_bar: progress_bar.progress((i+1)/len(df_f))
        p_table = Table([['Part No', format_part_no_v1(row['Part No'])], ['Description', Paragraph(str(row['Description']), bold_style_v1)]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
        p_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        
        l_vals = [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']]
        l_data = [[Paragraph('Line Location', location_header_style)] + [Paragraph(v, get_dynamic_location_style(v, 'Default')) for v in l_vals]]
        l_table = Table(l_data, colWidths=[4*cm] + [1.57*cm]*7, rowHeights=0.8*cm)
        l_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (1,0), (-1,0), colors.HexColor('#E9967A'))]))
        elements.extend([p_table, Spacer(1, 0.2*cm), l_table, Spacer(1, 0.8*cm)])
        if (i+1) % 4 == 0: elements.append(PageBreak())
    doc.build(elements); buffer.seek(0); return buffer

def generate_rack_labels_v2(df, progress_bar=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, margin=1.5*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY']
    for i, (_, row) in enumerate(df_f.iterrows()):
        if progress_bar: progress_bar.progress((i+1)/len(df_f))
        p_table = Table([['Part No', format_part_no_v2(row['Part No'])], ['Description', Paragraph(str(row['Description']), desc_style)]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        p_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        l_vals = [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']]
        l_data = [[Paragraph('Line Location', location_header_style)] + [Paragraph(v, get_dynamic_location_style(v, 'Default')) for v in l_vals]]
        l_table = Table(l_data, colWidths=[4*cm] + [1.57*cm]*7, rowHeights=0.9*cm)
        l_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (1,0), (-1,0), colors.HexColor('#ADD8E6'))]))
        elements.extend([p_table, Spacer(1, 0.3*cm), l_table, Spacer(1, 1.5*cm)])
        if (i+1) % 4 == 0: elements.append(PageBreak())
    doc.build(elements); buffer.seek(0); return buffer

def generate_bin_labels(df, mtm_models, progress_bar=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(10*cm, 15*cm), margin=0.2*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY']
    for i, (_, row) in enumerate(df_f.iterrows()):
        if progress_bar: progress_bar.progress((i+1)/len(df_f))
        t1 = Table([["Part No", Paragraph(str(row['Part No']), bin_bold_style)], ["Desc", Paragraph(str(row['Description']), bin_desc_style)], ["Qty/Bin", Paragraph(str(row['Qty/Bin']), bin_qty_style)]], colWidths=[3*cm, 6.5*cm], rowHeights=[1*cm, 1*cm, 0.7*cm])
        t1.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        
        qr_img = None
        if QR_AVAILABLE:
            qr_data = f"PN:{row['Part No']}\nLOC:{row['Level']}{row['Physical_Cell']}"
            qr_img = RLImage(BytesIO(qrcode.make(qr_data).tobytes()), width=3*cm, height=3*cm) # Simplified for example
            
        elements.extend([t1, Spacer(1, 1*cm), PageBreak()])
    doc.build(elements); buffer.seek(0); return buffer

# --- Main Application ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    
    output_type = st.sidebar.selectbox("Output Type", ["Rack Labels", "Bin Labels", "Rack List"])
    base_rack_id = st.sidebar.text_input("Infrastructure Prefix (e.g. R, TR)", "R")
    num_racks = st.sidebar.number_input("Racks per Station", 1, 10, 1)
    levels = st.sidebar.multiselect("Available Levels", ["A","B","C","D","E","F"], ["A","B","C"])
    num_cells_per_level = st.sidebar.number_input("Cells per Level", 1, 20, 6)

    uploaded_file = st.file_uploader("Upload Parts List", type=['xlsx', 'csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df.fillna('', inplace=True)
        cols = find_required_columns(df)
        
        if cols['Container']:
            unique_containers = sorted(df[cols['Container']].unique())
            bin_info_map = {}
            st.sidebar.markdown("---")
            st.sidebar.subheader("Container (Bin) Rules")
            
            for container in unique_containers:
                st.sidebar.markdown(f"**Settings for {container}**")
                dim = st.sidebar.text_input(f"Dimensions for {container}", "600x400", key=f"d_{container}")
                capacity = st.sidebar.number_input(f"Capacity (Parts/Cell) for {container}", 1, 10, 1, key=f"c_{container}")
                bin_info_map[container] = {'dims': parse_dimensions(dim), 'capacity': capacity}

            if st.button("🚀 Generate PDF Labels", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Setup Rack Configs based on User Input
                rack_configs = {f"Rack {i+1:02d}": {'levels': levels, 'cells_per_level': num_cells_per_level, 'dims': (1200, 1000)} for i in range(num_racks)}
                
                try:
                    df_res = automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status_text)
                    
                    pdf_buffer = None
                    if output_type == "Rack Labels":
                        fmt = st.selectbox("Choose Format", ["Multiple Parts", "Single Part"])
                        func = generate_rack_labels_v2 if fmt == "Single Part" else generate_rack_labels_v1
                        pdf_buffer = func(df_res, progress_bar)
                    elif output_type == "Bin Labels":
                        pdf_buffer = generate_bin_labels(df_res, [], progress_bar)
                    
                    if pdf_buffer:
                        st.success("✅ Automation Complete!")
                        st.download_button("📥 Download PDF", pdf_buffer.getvalue(), "AgiloSmartTag_Output.pdf")
                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
