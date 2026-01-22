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

# --- Style Definitions (The Original High-Detail Styles) ---
bold_style_v1 = ParagraphStyle(name='Bold_v1', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16, spaceBefore=5, spaceAfter=2)
bold_style_v2 = ParagraphStyle(name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16, spaceBefore=10, spaceAfter=15)
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16, spaceBefore=2, spaceAfter=2)
location_header_style = ParagraphStyle(name='LocationHeader', fontName='Helvetica', fontSize=16, alignment=TA_CENTER, leading=18)
location_value_style_base = ParagraphStyle(name='LocationValue_Base', fontName='Helvetica', alignment=TA_CENTER)

bin_bold_style = ParagraphStyle(name='BinBold', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, leading=14)
bin_desc_style = ParagraphStyle(name='BinDesc', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
bin_qty_style = ParagraphStyle(name='BinQty', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)

rl_header_style = ParagraphStyle(name='RL_Header', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)
rl_cell_left_style = ParagraphStyle(name='RL_Cell_Left', fontName='Helvetica', fontSize=9, alignment=TA_LEFT)

# --- Helper Functions ---
def parse_dims(dim_str):
    """Parses '600x400' into (600, 400)"""
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
    if column_type == 'Bus Model' and text_len > 5: f_size = 11
    elif column_type == 'Station No' and text_len > 4: f_size = 13
    return ParagraphStyle(name=f'Dyn_{text_len}', parent=location_value_style_base, fontSize=f_size, leading=f_size+2)

def find_required_columns(df):
    cols_map = {col.strip().upper(): col for col in df.columns}
    def find_col(patterns):
        for p in patterns:
            for k in cols_map:
                if p in k: return cols_map[k]
        return None
    return {
        'Part No': find_col(['PART NO', 'PART_NO']),
        'Description': find_col(['DESC']),
        'Bus Model': find_col(['BUS', 'MODEL']),
        'Station No': find_col(['STATION NO', 'ST_NO']),
        'Station Name': find_col(['STATION NAME', 'ST. NAME']),
        'Container': find_col(['CONTAINER']),
        'Qty/Bin': find_col(['QTY/BIN', 'QTY_BIN']),
        'Qty/Veh': find_col(['QTY/VEH', 'QTY_VEH']),
        'Zone': find_col(['ZONE', 'ABB ZONE'])
    }

# --- Infrastructure Automation Logic (The Core) ---
def automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status_text=None):
    cols = find_required_columns(df)
    df_p = df.copy()
    rename_map = {v: k for k, v in cols.items() if v}
    df_p.rename(columns=rename_map, inplace=True)

    # 1. Initialize physical cells with area calculation
    physical_slots = []
    for rack_name, config in sorted(rack_configs.items()):
        r_nums = re.findall(r'\d+', rack_name)
        r_val = r_nums[0].zfill(2) if r_nums else "01"
        
        # Calculate Physical Cell dimensions
        rack_w, rack_d = config['dims']
        cell_w = rack_w / config['cells_per_level'] if config['cells_per_level'] > 0 else 0
        cell_area = cell_w * rack_d
        
        for level in sorted(config['levels']):
            for i in range(1, config['cells_per_level'] + 1):
                physical_slots.append({
                    'Rack': base_rack_id, 'Rack No 1st': r_val[0], 'Rack No 2nd': r_val[1],
                    'Level': level, 'Physical_Cell': f"{i:02d}", 
                    'cell_area': cell_area, 'filled_parts': []
                })

    # 2. Assignment Logic: Comparing Bin Area to Cell Area
    final_output = []
    slot_idx = 0
    last_station = "N/A"

    for station_no, s_group in df_p.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Processing Station: {station_no}")
        last_station = station_no
        
        # Priority sort: Process parts with the largest containers first
        s_group['bin_area'] = s_group['Container'].map(lambda x: bin_info_map.get(x, {}).get('dims', (0,0))[0] * bin_info_map.get(x, {}).get('dims', (0,0))[1])
        parts_list = s_group.sort_values(by='bin_area', ascending=False).to_dict('records')

        for part in parts_list:
            if slot_idx >= len(physical_slots): break
            
            cont_type = str(part.get('Container', ''))
            b_info = bin_info_map.get(cont_type, {'dims': (0,0), 'capacity': 1})
            bin_area = b_info['dims'][0] * b_info['dims'][1]
            
            slot = physical_slots[slot_idx]
            
            # Calculate how many of these specific bins fit in the current physical cell
            # Formula: min(User Manual Cap, Floor(Cell Area / Bin Area))
            area_cap = int(slot['cell_area'] // bin_area) if bin_area > 0 else 1
            effective_cap = min(b_info['capacity'], area_cap)
            if effective_cap < 1: effective_cap = 1

            # Update part with location
            part.update({
                'Rack': slot['Rack'], 'Rack No 1st': slot['Rack No 1st'], 
                'Rack No 2nd': slot['Rack No 2nd'], 'Level': slot['Level'], 
                'Physical_Cell': slot['Physical_Cell']
            })
            final_output.append(part)
            
            # Move to next slot if this one is full
            slot['filled_parts'].append(part)
            if len(slot['filled_parts']) >= effective_cap:
                slot_idx += 1

    # 3. Fill the rest of the physical infrastructure with "EMPTY"
    while slot_idx < len(physical_slots):
        empty = {'Part No': 'EMPTY', 'Description': '', 'Bus Model': '', 'Station No': last_station}
        empty.update(physical_slots[slot_idx])
        final_output.append(empty)
        slot_idx += 1

    return pd.DataFrame(final_output)

# --- PDF Generation: Rack Labels ---
def generate_rack_labels(df, format_type, progress_bar=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, margin=1*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY']
    
    for i, (_, row) in enumerate(df_f.iterrows()):
        if progress_bar: progress_bar.progress((i+1)/len(df_f))
        
        if format_type == "Single Part":
            p_table = Table([['Part No', format_part_no_v2(row['Part No'])], ['Description', Paragraph(str(row['Description']), desc_style)]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        else:
            p_table = Table([['Part No', format_part_no_v1(row['Part No'])], ['Description', Paragraph(str(row['Description']), bold_style_v1)]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
        
        p_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        
        l_vals = [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']]
        l_data = [[Paragraph('Line Location', location_header_style)] + [Paragraph(v, get_dynamic_location_style(v, 'Default')) for v in l_vals]]
        l_table = Table(l_data, colWidths=[4*cm] + [1.57*cm]*7, rowHeights=0.9*cm)
        l_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('BACKGROUND', (1,0), (-1,0), colors.HexColor('#ADD8E6') if format_type=="Single Part" else colors.HexColor('#E9967A'))]))
        
        elements.extend([p_table, Spacer(1, 0.3*cm), l_table, Spacer(1, 1*cm)])
        if (i+1) % 4 == 0: elements.append(PageBreak())
        
    doc.build(elements); buffer.seek(0); return buffer

# --- PDF Generation: Professional Bin Labels (High Detail) ---
def generate_bin_labels(df, mtm_models, progress_bar=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(10*cm, 15*cm), margin=0.1*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY']

    def draw_border(canvas, doc):
        canvas.setStrokeColor(colors.black); canvas.setLineWidth(1.8)
        canvas.rect(0.2*cm, 7.5*cm, 9.6*cm, 7.2*cm)

    for i, (_, row) in enumerate(df_f.iterrows()):
        if progress_bar: progress_bar.progress((i+1)/len(df_f))
        
        # 1. Main Header Table
        t1 = Table([["Part No", Paragraph(str(row['Part No']), bin_bold_style)], ["Description", Paragraph(str(row['Description']), bin_desc_style)], ["Qty/Bin", Paragraph(str(row['Qty/Bin']), bin_qty_style)]], colWidths=[3*cm, 6.6*cm], rowHeights=[0.9*cm, 1.0*cm, 0.5*cm])
        t1.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1.2, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        
        # 2. Store Location Placeholder Table
        sl_inner = Table([[""]*7], colWidths=[0.94*cm]*7, rowHeights=[0.5*cm])
        sl_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black)]))
        sl_table = Table([["Store Location", sl_inner]], colWidths=[3*cm, 6.6*cm])
        sl_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))

        # 3. Line Location Table
        l_vals = [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']]
        ll_inner = Table([l_vals], colWidths=[0.94*cm]*7, rowHeights=[0.5*cm])
        ll_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('FONTSIZE', (0,0),(-1,-1), 8), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        ll_table = Table([["Line Location", ll_inner]], colWidths=[3*cm, 6.6*cm])
        ll_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))

        # 4. QR and MTM
        qr_img = None
        if QR_AVAILABLE:
            qr = qrcode.QRCode(box_size=10, border=2)
            qr.add_data(f"PN:{row['Part No']}\nLOC:{row['Level']}{row['Physical_Cell']}")
            qr.make(fit=True); img_io = BytesIO(); qr.make_image().save(img_io, format='PNG'); img_io.seek(0)
            qr_img = RLImage(img_io, width=2.5*cm, height=2.5*cm)

        mtm_qty = [str(row['Qty/Veh']) if str(row['Bus Model']).upper() == m.upper() else "" for m in mtm_models]
        mtm_t = Table([mtm_models, mtm_qty], colWidths=[1.1*cm]*len(mtm_models), rowHeights=[0.7*cm, 0.7*cm])
        mtm_t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTSIZE', (0,0), (-1,-1), 8)]))

        bottom = Table([[mtm_t, Spacer(1, 1*cm), qr_img]], colWidths=[4.5*cm, 1.5*cm, 3.5*cm])
        
        elements.extend([t1, sl_table, ll_table, Spacer(1, 0.3*cm), bottom, PageBreak()])
        
    doc.build(elements, onFirstPage=draw_border, onLaterPages=draw_border)
    buffer.seek(0); return buffer

# --- PDF Generation: Professional Rack List ---
def generate_rack_list_pdf(df, base_rack_id, footer_logo_path, progress_bar=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), margin=0.5*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY'].copy()

    for (st_no, r1), group in df_f.groupby(['Station No', 'Rack No 1st']):
        header = Table([["STATION NO", str(st_no), "RACK NO", f"{base_rack_id}{r1}"]], colWidths=[4*cm, 9*cm, 4*cm, 9*cm])
        header.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#8EAADB"))]))
        elements.append(header); elements.append(Spacer(1, 0.5*cm))

        data = [["S.NO", "PART NO", "DESCRIPTION", "CONTAINER", "QTY/BIN", "LOCATION"]]
        for idx, r in enumerate(group.to_dict('records')):
            loc = f"{r['Bus Model']}-{r['Station No']}-{r['Rack']}{r['Rack No 1st']}{r['Rack No 2nd']}-{r['Level']}{r['Physical_Cell']}"
            data.append([idx+1, r['Part No'], Paragraph(r['Description'], rl_cell_left_style), r['Container'], r['Qty/Bin'], loc])
        
        tm = Table(data, colWidths=[1.5*cm, 4*cm, 10*cm, 3*cm, 2*cm, 7*cm])
        tm.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.orange), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        elements.append(tm); elements.append(Spacer(1, 0.5*cm))

        footer = Table([[f"Date: {datetime.date.today()}", "Verified: ________________", "Designed by Agilomatrix"]], colWidths=[8*cm, 12*cm, 6*cm])
        elements.append(footer); elements.append(PageBreak())

    doc.build(elements); buffer.seek(0); return buffer

# --- Main Streamlit Application UI ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    
    # 1. Sidebar - Infrastructure
    st.sidebar.title("1. Infrastructure Setup")
    output_type = st.sidebar.selectbox("Output Type", ["Rack Labels", "Bin Labels", "Rack List"])
    base_prefix = st.sidebar.text_input("Infrastructure Prefix (e.g., R, TR)", "R")
    num_racks = st.sidebar.number_input("Racks per Station", 1, 10, 1)
    levels = st.sidebar.multiselect("Rack Levels", ["A","B","C","D","E","F"], ["A","B","C"])
    cells_per_lvl = st.sidebar.number_input("Cells per Level", 1, 20, 6)
    
    # 2. Sidebar - Rack Dimensions
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Rack Dimensions (mm)")
    rack_w = st.sidebar.number_input("Total Rack Width", 1200)
    rack_d = st.sidebar.number_input("Total Rack Depth", 1000)

    # 3. Sidebar - MTM Models
    mtm_models = []
    if output_type == "Bin Labels":
        st.sidebar.markdown("---")
        st.sidebar.subheader("3. MTM Models")
        m1, m2, m3 = st.sidebar.text_input("M1", "7M"), st.sidebar.text_input("M2", "9M"), st.sidebar.text_input("M3", "12M")
        mtm_models = [m for m in [m1, m2, m3] if m]

    uploaded_file = st.file_uploader("Upload Parts Data", type=['xlsx', 'csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df.fillna('', inplace=True)
        cols = find_required_columns(df)
        
        if cols['Container']:
            unique_containers = sorted(df[cols['Container']].unique())
            bin_info_map = {}
            st.sidebar.markdown("---")
            st.sidebar.subheader("4. Container Dimensions & Rules")
            
            for cont in unique_containers:
                st.sidebar.write(f"**{cont}**")
                dim_str = st.sidebar.text_input(f"Dims (WxD)", "600x400", key=f"d_{cont}")
                cap = st.sidebar.number_input(f"Manual Max Capacity", 1, 10, 1, key=f"c_{cont}")
                bin_info_map[cont] = {'dims': parse_dims(dim_str), 'capacity': cap}

            if st.button("🚀 Run Full Studio Automation", type="primary"):
                progress = st.progress(0); status = st.empty()
                
                # Setup Configuration
                rack_configs = {f"Rack {i+1:02d}": {'levels': levels, 'cells_per_level': cells_per_lvl, 'dims': (rack_w, rack_d)} for i in range(num_racks)}
                
                # Execute Logic
                df_results = automate_location_assignment(df, base_prefix, rack_configs, bin_info_map, status)
                
                # Calculate Cell Dims for UI transparency
                cell_w_calc = rack_w / cells_per_lvl
                st.info(f"Physical Cell Size: {cell_w_calc:.1f}mm (W) x {rack_d:.1f}mm (D)")

                pdf = None
                if output_type == "Rack Labels":
                    fmt = st.selectbox("Format", ["Single Part", "Multiple Parts"])
                    pdf = generate_rack_labels(df_results, fmt, progress)
                elif output_type == "Bin Labels":
                    pdf = generate_bin_labels(df_results, mtm_models, progress)
                elif output_type == "Rack List":
                    pdf = generate_rack_list_pdf(df_results, base_prefix, "Image.png", progress)
                
                if pdf:
                    st.success("PDF Studio Generation Complete!")
                    st.download_button("📥 Download Result", pdf.getvalue(), f"Studio_Output_{datetime.date.today()}.pdf")
        else:
            st.error("Missing Container column in the uploaded file.")

if __name__ == "__main__":
    main()
