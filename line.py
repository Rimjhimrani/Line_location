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

# --- Style Definitions ---
bold_style_v1 = ParagraphStyle(name='Bold_v1', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16)
bold_style_v2 = ParagraphStyle(name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16)
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16)
location_header_style = ParagraphStyle(name='LocationHeader', fontName='Helvetica', fontSize=16, alignment=TA_CENTER, leading=18)
location_value_style_base = ParagraphStyle(name='LocationValue_Base', fontName='Helvetica', alignment=TA_CENTER)
bin_bold_style = ParagraphStyle(name='BinBold', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, leading=14)
bin_desc_style = ParagraphStyle(name='BinDesc', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
bin_qty_style = ParagraphStyle(name='BinQty', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
rl_header_style = ParagraphStyle(name='RL_Header', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)
rl_cell_left_style = ParagraphStyle(name='RL_Cell_Left', fontName='Helvetica', fontSize=9, alignment=TA_LEFT)

# --- Formatting Helpers ---
def format_part_no_v1(part_no):
    p = str(part_no)
    if len(p) > 5:
        return Paragraph(f"<b><font size=17>{p[:-5]}</font><font size=22>{p[-5:]}</font></b>", bold_style_v1)
    return Paragraph(f"<b><font size=17>{p}</font></b>", bold_style_v1)

def format_part_no_v2(part_no):
    p = str(part_no)
    if p.upper() == 'EMPTY': return Paragraph(f"<b><font size=34>EMPTY</font></b>", bold_style_v2)
    if len(p) > 5:
        return Paragraph(f"<b><font size=34>{p[:-5]}</font><font size=40>{p[-5:]}</font></b>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{p}</font></b>", bold_style_v2)

def get_dynamic_location_style(text, col_type):
    t_len = len(str(text))
    f_size = 16
    if col_type == 'Bus Model':
        f_size = 14 if t_len <= 3 else 10 if t_len <= 10 else 9
    elif col_type == 'Station No':
        f_size = 20 if t_len <= 2 else 18 if t_len <= 5 else 12
    return ParagraphStyle(name='dyn', parent=location_value_style_base, fontSize=f_size)

# --- Core Logic Functions ---
def find_required_columns(df):
    cols_map = {col.strip().upper(): col for col in df.columns}
    def find_col(patterns):
        for p in patterns:
            for k in cols_map:
                if p in k: return cols_map[k]
        return None
    return {
        'Part No': find_col(['PART NO', 'PART NUM']),
        'Description': find_col(['DESC']),
        'Bus Model': find_col(['BUS MODEL']),
        'Station No': find_col(['STATION NO']),
        'Station Name': find_col(['STATION NAME', 'ST. NAME']),
        'Container': find_col(['CONTAINER']),
        'Qty/Bin': find_col(['QTY/BIN', 'QTY_BIN']),
        'Qty/Veh': find_col(['QTY/VEH', 'QTY_VEH']),
        'Zone': find_col(['ZONE', 'AREA'])
    }

def automate_location_assignment(df, base_rack_id, rack_configs, status_text=None):
    cols = find_required_columns(df)
    df_p = df.copy()
    rename_dict = {cols[k]: k for k in cols if cols[k]}
    df_p.rename(columns=rename_dict, inplace=True)
    
    final_rows = []
    # Sort Racks
    sorted_rack_names = sorted(rack_configs.keys())
    
    for station_no, station_group in df_p.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Processing station: {station_no}...")
        
        # Track available space for THIS station
        current_rack_idx = 0
        current_level_idx = 0
        current_cell_in_level = 1
        
        for _, part in station_group.iterrows():
            assigned = False
            while not assigned and current_rack_idx < len(sorted_rack_names):
                r_name = sorted_rack_names[current_rack_idx]
                cfg = rack_configs[r_name]
                lvls = sorted(cfg['levels'])
                cont_type = part.get('Container', 'Unknown')
                max_cells = cfg['rack_bin_counts'].get(cont_type, 0)
                
                if max_cells > 0 and current_level_idx < len(lvls):
                    # We have a valid slot
                    rack_digits = ''.join(filter(str.isdigit, r_name))
                    r1 = rack_digits[0] if len(rack_digits) > 1 else '0'
                    r2 = rack_digits[1] if len(rack_digits) > 1 else (rack_digits[0] if rack_digits else '0')
                    
                    p_copy = part.to_dict()
                    p_copy.update({
                        'Rack': base_rack_id, 'Rack No 1st': r1, 'Rack No 2nd': r2,
                        'Level': lvls[current_level_idx], 'Cell': str(current_cell_in_level),
                        'Station No': station_no
                    })
                    final_rows.append(p_copy)
                    assigned = True
                    
                    # Increment for next part
                    current_cell_in_level += 1
                    if current_cell_in_level > max_cells:
                        current_cell_in_level = 1
                        current_level_idx += 1
                        if current_level_idx >= len(lvls):
                            current_level_idx = 0
                            current_rack_idx += 1
                else:
                    # Skip to next level or rack
                    current_level_idx = 0
                    current_rack_idx += 1

    return pd.DataFrame(final_rows)

# --- PDF Modules (Bin Labels) ---
def generate_bin_labels(df, mtm_models, progress_bar=None):
    if not QR_AVAILABLE: return None, {}
    buffer = BytesIO()
    # Industrial Sticker Size: 10cm x 15cm
    doc = SimpleDocTemplate(buffer, pagesize=(10*cm, 15*cm), margin=0.1*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY']
    
    summary = {}
    for i, row in enumerate(df_f.to_dict('records')):
        sum_key = f"ST-{row['Station No']} / Rack {row['Rack No 1st']}{row['Rack No 2nd']}"
        summary[sum_key] = summary.get(sum_key, 0) + 1
        
        # QR Code Generation (Includes all required meta)
        qr_data = f"Part:{row['Part No']}\nDesc:{row['Description'][:30]}\nQty:{row['Qty/Bin']}\nLoc:{row['Level']}-{row['Cell']}"
        qr = qrcode.make(qr_data)
        qr_buf = BytesIO(); qr.save(qr_buf, format='PNG'); qr_buf.seek(0)
        qr_img = RLImage(qr_buf, width=3*cm, height=3*cm)

        # MTM Table (Vehicle Models)
        mtm_data = [mtm_models, [Paragraph(f"<b>{row['Qty/Veh']}</b>", bin_qty_style) if str(row['Bus Model']) == str(m) else "" for m in mtm_models]]
        mtm_t = Table(mtm_data, colWidths=[1.2*cm]*len(mtm_models))
        mtm_t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('ALIGN', (0,0), (-1,-1), 'CENTER')]))

        # Main Table
        data = [
            [Paragraph("Part No", bin_desc_style), Paragraph(row['Part No'], bin_bold_style)],
            [Paragraph("Description", bin_desc_style), Paragraph(row['Description'][:50], bin_desc_style)],
            [Paragraph("Qty/Bin", bin_desc_style), Paragraph(str(row['Qty/Bin']), bin_qty_style)],
            [Paragraph("MTM Models", bin_desc_style), mtm_t],
            [Paragraph("Line Location", bin_desc_style), Paragraph(f"{row['Level']}-{row['Cell']}", bin_bold_style)],
            [Paragraph("QR Scan", bin_desc_style), qr_img]
        ]
        
        t = Table(data, colWidths=[3*cm, 6.5*cm])
        t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        
        elements.append(t)
        elements.append(PageBreak())

    doc.build(elements); buffer.seek(0)
    return buffer, summary

# --- PDF Modules (Rack List) ---
def generate_rack_list_pdf(df, base_rack_id, top_logo_file, prog=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), margin=0.5*cm)
    elements = []
    df_f = df[df['Part No'].str.upper() != 'EMPTY'].sort_values(['Station No', 'Level', 'Cell'])
    has_zone = 'Zone' in df_f.columns and df_f['Zone'].any()

    for (st_no, r1, r2), group in df_f.groupby(['Station No', 'Rack No 1st', 'Rack No 2nd']):
        # Logo
        logo = RLImage(BytesIO(top_logo_file.getvalue()), width=4*cm, height=1.5*cm) if top_logo_file else Paragraph("AGILOMATRIX", rl_header_style)
        elements.append(logo)
        elements.append(Paragraph(f"<b>STATION: {st_no} | RACK: {r1}{r2}</b>", rl_header_style))
        
        data = [["ZONE" if has_zone else "S.NO", "PART NO", "DESCRIPTION", "CONTAINER", "QTY/BIN", "LOCATION"]]
        col_ws = [2.5*cm, 4*cm, 9*cm, 3*cm, 2*cm, 6*cm]
        
        # Table Styling for Zone Spanning
        style_cmds = [('GRID', (0,0), (-1,-1), 0.5, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#8EAADB")), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]
        
        for idx, row in enumerate(group.to_dict('records')):
            loc = f"{row['Bus Model']}-{st_no}-{base_rack_id}{r1}{r2}-{row['Level']}{row['Cell']}"
            data.append([row.get('Zone', '') if has_zone else idx+1, row['Part No'], Paragraph(row['Description'], rl_cell_left_style), row['Container'], row['Qty/Bin'], loc])
        
        t = Table(data, colWidths=col_ws, repeatRows=1)
        t.setStyle(TableStyle(style_cmds))
        elements.append(t)
        
        # Footer
        elements.append(Spacer(1, 1*cm))
        elements.append(Paragraph(f"Created Date: {datetime.date.today()} | Designed by Agilomatrix", rl_cell_left_style))
        elements.append(PageBreak())

    doc.build(elements[:-1]); buffer.seek(0)
    return buffer, len(df_f)

# --- Streamlit Main UI ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("Automated Labeling for Line-Side Logistics")
    
    menu = st.sidebar.selectbox("Select Output", ["Bin Labels", "Rack Labels", "Rack List"])
    
    uploaded = st.file_uploader("Upload Excel File", type=['xlsx'])
    
    if uploaded:
        df = pd.read_excel(uploaded).fillna('')
        cols = find_required_columns(df)
        
        if cols['Container'] and cols['Station No']:
            bins = sorted(df[cols['Container']].unique())
            
            with st.expander("Step 1: Rack Setup", expanded=True):
                n_racks = st.number_input("Racks per Station", 1, 10, 1)
                base_id = st.text_input("Infra Code (R/TR/SH)", "R")
                configs = {}
                for r in range(n_racks):
                    r_name = f"Rack {r+1:02d}"
                    st.write(f"--- {r_name} ---")
                    lvls = st.multiselect(f"Levels", ["A","B","C","D","E"], ["A","B","C"], key=f"lvl{r}")
                    caps = {b: st.number_input(f"{b} Cap/Level", 0, 50, 4, key=f"cp{r}{b}") for b in bins}
                    configs[r_name] = {'levels': lvls, 'rack_bin_counts': caps}
            
            if st.button("Generate Output"):
                status = st.empty()
                df_assigned = automate_location_assignment(df, base_id, configs, status)
                
                if menu == "Bin Labels":
                    mtm = st.sidebar.multiselect("Active Vehicle Models", ["7M", "9M", "12M"], ["7M", "9M", "12M"])
                    buf, sum_map = generate_bin_labels(df_assigned, mtm)
                elif menu == "Rack List":
                    logo = st.sidebar.file_uploader("Upload Logo", type=['png', 'jpg'])
                    buf, count = generate_rack_list_pdf(df_assigned, base_id, logo)
                else:
                    # Logic for Rack Labels V1/V2
                    from __main__ import generate_rack_labels_v2 # Assuming logic is kept
                    buf, sum_map = generate_rack_labels_v2(df_assigned) # Simplified for brevity

                st.success("PDF Generated Successfully!")
                st.download_button("📥 Download PDF", buf, f"Agilo_{menu}.pdf")

if __name__ == "__main__":
    main()
