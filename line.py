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

# --- Style Definitions (The Professional Set) ---
bold_style_v1 = ParagraphStyle(name='Bold_v1', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16, spaceBefore=5, spaceAfter=2)
bold_style_v2 = ParagraphStyle(name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16, spaceBefore=10, spaceAfter=15)
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16, spaceBefore=2, spaceAfter=2)
location_header_style = ParagraphStyle(name='LocationHeader', fontName='Helvetica', fontSize=16, alignment=TA_CENTER, leading=18)
location_value_style_base = ParagraphStyle(name='LocationValue_Base', fontName='Helvetica', alignment=TA_CENTER)
bin_bold_style = ParagraphStyle(name='BinBold', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, leading=14)
bin_desc_style = ParagraphStyle(name='BinDesc', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
bin_qty_style = ParagraphStyle(name='BinQty', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
rl_header_style = ParagraphStyle(name='RL_Header', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)
rl_cell_style = ParagraphStyle(name='RL_Cell', fontName='Helvetica', fontSize=9, alignment=TA_CENTER)
rl_cell_left_style = ParagraphStyle(name='RL_Cell_Left', fontName='Helvetica', fontSize=9, alignment=TA_LEFT)

# --- Helper Parsers ---
def parse_dims(dim_str):
    """Parses '1200x1000' into (1200, 1000)"""
    if not dim_str: return (0, 0)
    nums = re.findall(r'\d+', dim_str)
    if len(nums) >= 2: return (int(nums[0]), int(nums[1]))
    return (0, 0)

# --- Formatting: Part Numbers ---
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

# --- Formatting: Location Cells Dynamic Font ---
def get_dynamic_location_style(text, column_type):
    text_len = len(str(text))
    font_size, leading = 16, 18
    if column_type == 'Bus Model':
        font_size = 14 if text_len <= 3 else 12 if text_len <= 5 else 10
    elif column_type == 'Station No':
        font_size = 20 if text_len <= 2 else 18 if text_len <= 5 else 12
    else:
        font_size = 16 if text_len <= 2 else 14 if text_len <= 4 else 12
    return ParagraphStyle(name=f'Dyn_{text_len}', parent=location_value_style_base, fontSize=font_size, leading=leading)

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

# --- Infrastructure Automation Core ---
def automate_location_assignment(df, base_rack_id, rack_configs, bin_info_map, status_text=None):
    cols = find_required_columns(df)
    df_p = df.copy()
    rename_map = {v: k for k, v in cols.items() if v}
    df_p.rename(columns=rename_map, inplace=True)

    # 1. Create Physical Slots with Dimensions
    available_slots = []
    for rack_name, config in sorted(rack_configs.items()):
        r_nums = re.findall(r'\d+', rack_name)
        r_val = r_nums[0].zfill(2) if r_nums else "01"
        
        # Calculate Cell Dimensions
        rack_w, rack_d = config['dims']
        cell_w = rack_w / config['cells_per_level'] if config['cells_per_level'] > 0 else 0
        cell_d = rack_d # Standard depth
        
        for level in sorted(config['levels']):
            for i in range(1, config['cells_per_level'] + 1):
                available_slots.append({
                    'Rack': base_rack_id, 'Rack No 1st': r_val[0], 'Rack No 2nd': r_val[1],
                    'Level': level, 'Physical_Cell': f"{i:02d}", 
                    'cell_w': cell_w, 'cell_d': cell_d, 'filled_count': 0
                })

    # 2. Assign Parts based on Sidebar Capacity
    final_data = []
    slot_idx = 0
    last_station = "N/A"

    for station_no, s_group in df_p.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Allocating Station: {station_no}...")
        last_station = station_no
        parts = s_group.to_dict('records')

        for part in parts:
            if slot_idx >= len(available_slots): break
            
            cont_type = str(part.get('Container', ''))
            capacity = bin_info_map.get(cont_type, {}).get('capacity', 1)
            
            slot = available_slots[slot_idx]
            part.update(slot)
            final_data.append(part)
            
            slot['filled_count'] += 1
            if slot['filled_count'] >= capacity:
                slot_idx += 1

    # 3. Fill Empty
    while slot_idx < len(available_slots):
        empty = {'Part No': 'EMPTY', 'Description': '', 'Bus Model': '', 'Station No': last_station}
        empty.update(available_slots[slot_idx])
        final_data.append(empty)
        slot_idx += 1

    return pd.DataFrame(final_data)

# --- PDF Gen: Rack Labels ---
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
        l_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (1,0), (-1,0), colors.HexColor('#ADD8E6') if format_type=="Single Part" else colors.HexColor('#E9967A'))]))
        
        elements.extend([p_table, Spacer(1, 0.3*cm), l_table, Spacer(1, 1*cm)])
        if (i+1) % 4 == 0: elements.append(PageBreak())
        
    doc.build(elements); buffer.seek(0); return buffer

# --- PDF Gen: Full Bin Labels (Stickers) ---
def generate_bin_labels(df, mtm_models, progress_bar=None):
    if not QR_AVAILABLE: return None
    buffer = BytesIO()
    # 10x15cm Sticker
    doc = SimpleDocTemplate(buffer, pagesize=(10*cm, 15*cm), margin=0.1*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY']

    def draw_border(canvas, doc):
        canvas.setStrokeColor(colors.black)
        canvas.setLineWidth(1.8)
        canvas.rect(0.2*cm, 7.5*cm, 9.6*cm, 7.2*cm) # Border logic

    for i, (_, row) in enumerate(df_f.iterrows()):
        if progress_bar: progress_bar.progress((i+1)/len(df_f))
        
        # 1. Main Table
        t1 = Table([["Part No", Paragraph(str(row['Part No']), bin_bold_style)], ["Description", Paragraph(str(row['Description']), bin_desc_style)], ["Qty/Bin", Paragraph(str(row['Qty/Bin']), bin_qty_style)]], colWidths=[3*cm, 6.6*cm], rowHeights=[0.9*cm, 1.0*cm, 0.5*cm])
        t1.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1.2, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        
        # 2. Store Location (Placeholder logic for columns)
        sl_vals = [""] * 7 # Generic store placeholders
        sl_inner = Table([sl_vals], colWidths=[0.94*cm]*7, rowHeights=[0.5*cm])
        sl_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('FONTSIZE', (0,0),(-1,-1), 7)]))
        sl_table = Table([["Store Location", sl_inner]], colWidths=[3*cm, 6.6*cm])
        sl_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))

        # 3. Line Location (Automation data)
        ll_vals = [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']]
        ll_inner = Table([ll_vals], colWidths=[0.94*cm]*7, rowHeights=[0.5*cm])
        ll_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('FONTSIZE', (0,0),(-1,-1), 8), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        ll_table = Table([["Line Location", ll_inner]], colWidths=[3*cm, 6.6*cm])
        ll_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))

        # 4. MTM + QR
        qr_data = f"PN:{row['Part No']}\nLL:{''.join(ll_vals)}"
        qr_gen = qrcode.QRCode(box_size=10, border=2)
        qr_gen.add_data(qr_data); qr_gen.make(fit=True)
        img_b = BytesIO(); qr_gen.make_image().save(img_b, format='PNG'); img_b.seek(0)
        qr_rl = RLImage(img_b, width=2.5*cm, height=2.5*cm)

        mtm_qty = [str(row['Qty/Veh']) if str(row['Bus Model']).upper() == m.upper() else "" for m in mtm_models]
        mtm_t = Table([mtm_models, mtm_qty], colWidths=[1.1*cm]*len(mtm_models), rowHeights=[0.7*cm, 0.7*cm])
        mtm_t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTSIZE', (0,0), (-1,-1), 8)]))

        bottom = Table([[mtm_t, Spacer(1, 1*cm), qr_rl]], colWidths=[4*cm, 2*cm, 3*cm])
        
        elements.extend([t1, sl_table, ll_table, Spacer(1, 0.3*cm), bottom, PageBreak()])
        
    doc.build(elements, onFirstPage=draw_border, onLaterPages=draw_border)
    buffer.seek(0); return buffer

# --- PDF Gen: Professional Rack List ---
def generate_rack_list_pdf(df, base_rack_id, top_logo_file, footer_logo_path, progress_bar=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), margin=0.5*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY'].copy()

    for (st_no, r1), group in df_f.groupby(['Station No', 'Rack No 1st']):
        # Header Table (Document Ref + Logo)
        logo = RLImage(top_logo_file, width=4*cm, height=1.2*cm) if top_logo_file else ""
        header_table = Table([["STATION NO", str(st_no), "RACK NO", f"{base_rack_id}{r1}", logo]], colWidths=[3*cm, 8*cm, 3*cm, 8*cm, 4*cm])
        header_table.setStyle(TableStyle([('GRID', (0,0), (3,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#8EAADB"))]))
        elements.append(header_table); elements.append(Spacer(1, 0.3*cm))

        # Main Data Table
        data = [["S.NO", "PART NO", "DESCRIPTION", "CONTAINER", "QTY/BIN", "LOCATION"]]
        for idx, r in enumerate(group.to_dict('records')):
            loc = f"{r['Bus Model']}-{r['Station No']}-{r['Rack']}{r['Rack No 1st']}{r['Rack No 2nd']}-{r['Level']}{r['Physical_Cell']}"
            data.append([idx+1, r['Part No'], Paragraph(r['Description'], rl_cell_left_style), r['Container'], r['Qty/Bin'], loc])
        
        tm = Table(data, colWidths=[1.5*cm, 4*cm, 10*cm, 3*cm, 2*cm, 7*cm])
        tm.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.orange), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        elements.append(tm)

        # Footer
        elements.append(Spacer(1, 0.5*cm))
        footer_data = [[f"Creation Date: {datetime.date.today()}", "Verified by: ________________", "Designed by:", RLImage(footer_logo_path, width=3*cm, height=0.8*cm) if os.path.exists(footer_logo_path) else ""]]
        tf = Table(footer_data, colWidths=[7*cm, 10*cm, 3*cm, 4*cm])
        elements.append(tf); elements.append(PageBreak())

    doc.build(elements); buffer.seek(0); return buffer

# --- Main App ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    
    st.sidebar.title("1. Infrastructure Setup")
    output_type = st.sidebar.selectbox("Choose Output:", ["Rack Labels", "Bin Labels", "Rack List"])
    base_prefix = st.sidebar.text_input("Prefix (e.g., R, TR)", "R")
    num_racks = st.sidebar.number_input("Racks per Station", 1, 10, 1)
    levels = st.sidebar.multiselect("Rack Levels", ["A","B","C","D","E","F"], ["A","B","C"])
    cells_per_lvl = st.sidebar.number_input("Cells per Level", 1, 20, 6)
    
    # Cell Dimension Inputs
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Rack Dimensions (mm)")
    rack_w = st.sidebar.number_input("Rack Total Width", 1200)
    rack_d = st.sidebar.number_input("Rack Total Depth", 1000)

    mtm_models = []
    if output_type == "Bin Labels":
        st.sidebar.markdown("---")
        st.sidebar.subheader("3. Vehicle Models (MTM)")
        m1, m2, m3 = st.sidebar.text_input("M1", "7M"), st.sidebar.text_input("M2", "9M"), st.sidebar.text_input("M3", "12M")
        mtm_models = [m for m in [m1, m2, m3] if m]

    uploaded_file = st.file_uploader("Upload Data", type=['xlsx', 'csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df.fillna('', inplace=True)
        cols = find_required_columns(df)
        
        if cols['Container']:
            unique_containers = sorted(df[cols['Container']].unique())
            bin_info_map = {}
            st.sidebar.markdown("---")
            st.sidebar.subheader("4. Container Capacity Rules")
            for cont in unique_containers:
                st.sidebar.write(f"**{cont}**")
                cap = st.sidebar.number_input(f"Parts per Cell", 1, 10, 1, key=f"c_{cont}")
                bin_info_map[cont] = {'capacity': cap}

            if st.button("🚀 Run Full Automation", type="primary"):
                progress = st.progress(0); status = st.empty()
                
                # Setup Rack Config
                rack_configs = {f"Rack {i+1:02d}": {'levels': levels, 'cells_per_level': cells_per_lvl, 'dims': (rack_w, rack_d)} for i in range(num_racks)}
                
                df_results = automate_location_assignment(df, base_prefix, rack_configs, bin_info_map, status)
                
                # Show Calculated Cell Dims for user
                cell_w = rack_w / cells_per_lvl
                st.info(f"Calculated Physical Cell Dimensions: {cell_w:.1f}mm Width x {rack_d:.1f}mm Depth")

                pdf = None
                if output_type == "Rack Labels":
                    fmt = st.selectbox("Format", ["Single Part", "Multiple Parts"])
                    pdf = generate_rack_labels(df_results, fmt, progress)
                elif output_type == "Bin Labels":
                    pdf = generate_bin_labels(df_results, mtm_models, progress)
                elif output_type == "Rack List":
                    pdf = generate_rack_list_pdf(df_results, base_prefix, None, "Image.png", progress)
                
                if pdf:
                    st.success("PDF Generated!")
                    st.download_button("📥 Download", pdf.getvalue(), f"Studio_{datetime.date.today()}.pdf")
        else:
            st.error("Missing Container Column.")

if __name__ == "__main__":
    main()
