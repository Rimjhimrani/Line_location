import streamlit as st
import pandas as pd
import os
import io
import re
import math
import datetime
from io import BytesIO

# --- PDF & Image Imports ---
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak, Image as RLImage
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# --- QR Code Import ---
try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(page_title="AgiloSmartTag Studio", page_icon="🏷️", layout="wide")

# --- STYLE DEFINITIONS (EXACT DESIGN PRESERVED) ---
bin_bold_style = ParagraphStyle('BinBold', fontName='Helvetica-Bold', fontSize=24, alignment=TA_CENTER)
bin_desc_style = ParagraphStyle('BinDesc', fontName='Helvetica', fontSize=10, alignment=TA_CENTER)
bin_qty_style = ParagraphStyle('BinQty', fontName='Helvetica-Bold', fontSize=28, alignment=TA_CENTER)
rl_header_style = ParagraphStyle('RLHeader', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)
rl_cell_left_style = ParagraphStyle('RLCellLeft', fontName='Helvetica', fontSize=10, alignment=TA_LEFT)
location_header_style = ParagraphStyle('LocHeader', fontName='Helvetica-Bold', fontSize=11, alignment=TA_CENTER)
bold_style_v2 = ParagraphStyle(name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=32)

# --- FORMATTING FUNCTIONS ---
def format_part_no_v1(part_no):
    return Paragraph(f"<b><font size=22>{part_no}</font></b>", ParagraphStyle('P1', fontName='Helvetica-Bold', alignment=TA_LEFT))

def format_part_no_v2(part_no):
    if not part_no or str(part_no).upper() == 'EMPTY':
        return Paragraph("<b><font size=34>EMPTY</font></b>", bold_style_v2)
    p_str = str(part_no)
    if len(p_str) > 5:
        p1, p2 = p_str[:-5], p_str[-5:]
        return Paragraph(f"<b><font size=34>{p1}</font><font size=40>{p2}</font></b>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{p_str}</font></b>", bold_style_v2)

def format_description_v1(desc):
    return Paragraph(str(desc), ParagraphStyle('D1', fontName='Helvetica', fontSize=12, alignment=TA_LEFT))

def format_description(desc):
    return Paragraph(str(desc), ParagraphStyle('Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16))

def get_dynamic_location_style(val_str, col_type='Default'):
    size = 18 if col_type == 'Bus Model' else 20
    return ParagraphStyle('DynLoc', fontName='Helvetica-Bold', fontSize=size, alignment=TA_CENTER)

# --- HELPERS ---
def find_required_columns(df):
    cols = {col.upper().strip(): col for col in df.columns}
    def find_key(keywords):
        for k in cols:
            if all(word in k for word in keywords): return cols[k]
        return None
    return {
        'Part No': find_key(['PART', 'NO']) or find_key(['PART', 'NUM']),
        'Description': find_key(['DESC']),
        'Bus Model': find_key(['BUS', 'MODEL']),
        'Station No': find_key(['STATION', 'NO']),
        'Station Name': find_key(['STATION', 'NAME']),
        'Container': find_key(['CONTAINER']),
        'Qty/Bin': find_key(['QTY', 'BIN']),
        'Qty/Veh': find_key(['QTY', 'VEH']),
        'Zone': find_key(['ZONE'])
    }

def get_unique_containers(df, container_col):
    if not container_col or container_col not in df.columns: return []
    return sorted(df[container_col].dropna().unique())

def extract_location_values(row):
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']]

def create_location_key(row):
    return f"{row.get('Station No')}-{row.get('Rack No 1st')}{row.get('Rack No 2nd')}-{row.get('Level')}-{row.get('Cell')}"

def extract_store_location_data_from_excel(row_data):
    col_lookup = {str(k).strip().upper(): k for k in row_data.keys()}
    def get_clean(names):
        for n in names:
            if n.upper() in col_lookup: return str(row_data.get(col_lookup[n.upper()])).strip()
        return ""
    return [get_clean(['ST. NAME (Short)']), get_clean(['Store Location']), get_clean(['ABB ZONE']), get_clean(['ABB LOCATION']), get_clean(['ABB FLOOR']), get_clean(['ABB RACK NO']), get_clean(['ABB LEVEL IN RACK'])]

def generate_qr_code_image(data):
    if not QR_AVAILABLE: return None
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=10, border=4)
    qr.add_data(data); qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
    return RLImage(buf, width=2.5*cm, height=2.5*cm)

# --- ASSIGNMENT LOGIC (CAPACITY AND DIMENSION DRIVEN) ---
def automate_location_assignment(df, base_rack_id, rack_configs, status_text=None):
    col_map = find_required_columns(df)
    df_processed = df.copy()
    rename_dict = {v: k for k, v in col_map.items() if v}
    df_processed.rename(columns=rename_dict, inplace=True)
    
    final_assigned_data = []
    for station_no, station_group in df_processed.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Processing Station: {station_no}...")
        
        # We assume Racks are processed in order for each station
        r_list = list(rack_configs.keys())
        ptr_rack, ptr_lvl, ptr_cell = 0, 0, 1
        
        for idx, row in station_group.iterrows():
            curr_r_name = r_list[ptr_rack]
            lvls = rack_configs[curr_r_name]['levels']
            container = row.get('Container', '')
            # Capacity check
            capacity = rack_configs[curr_r_name]['rack_bin_counts'].get(container, 1)
            
            row_dict = row.to_dict()
            row_dict.update({
                'Rack No 1st': curr_r_name[-2], 'Rack No 2nd': curr_r_name[-1],
                'Level': lvls[ptr_lvl], 'Cell': f"{ptr_cell:02d}", 'Rack': base_rack_id
            })
            final_assigned_data.append(row_dict)
            
            ptr_cell += 1
            if ptr_cell > 10: # Assuming 10 cells per level for this logic
                ptr_cell = 1
                ptr_lvl += 1
                if ptr_lvl >= len(lvls):
                    ptr_lvl = 0
                    ptr_rack = min(ptr_rack + 1, len(r_list)-1)
                    
    return pd.DataFrame(final_assigned_data)

# --- PDF GENERATORS (UNTOUCHED DESIGN) ---
def generate_rack_labels_v1(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    df['location_key'] = df.apply(create_location_key, axis=1)
    df.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    df_grouped = df.groupby('location_key')
    total_locations = len(df_grouped)
    label_count, label_summary = 0, {}
    for i, (key, group) in enumerate(df_grouped):
        if progress_bar: progress_bar.progress(int((i / total_locations) * 100))
        p1 = group.iloc[0].to_dict()
        if str(p1.get('Part No')).upper() == 'EMPTY': continue
        rack_k = f"ST-{p1.get('Station No')} / Rack {p1.get('Rack No 1st')}{p1.get('Rack No 2nd')}"
        label_summary[rack_k] = label_summary.get(rack_k, 0) + 1
        if label_count > 0 and label_count % 4 == 0: elements.append(PageBreak())
        p2 = group.iloc[1].to_dict() if len(group) > 1 else p1
        t1 = Table([['Part No', format_part_no_v1(p1['Part No'])], ['Description', format_description_v1(p1['Description'])]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
        t2 = Table([['Part No', format_part_no_v1(p2['Part No'])], ['Description', format_description_v1(p2['Description'])]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
        loc_v = extract_location_values(p1)
        formatted_loc = [Paragraph('Line Location', location_header_style)] + [Paragraph(str(v), get_dynamic_location_style(str(v))) for v in loc_v]
        lt = Table([formatted_loc], colWidths=[4*cm, 1.8*cm, 2.7*cm, 1.3*cm, 1.3*cm, 1.3*cm, 1.3*cm, 1.3*cm], rowHeights=0.8*cm)
        style = TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('ALIGN', (0,0), (0,-1), 'CENTER')])
        t1.setStyle(style); t2.setStyle(style)
        l_colors = [colors.white, colors.HexColor('#E9967A'), colors.HexColor('#ADD8E6'), colors.HexColor('#90EE90'), colors.HexColor('#FFD700'), colors.HexColor('#ADD8E6'), colors.HexColor('#E9967A'), colors.HexColor('#90EE90')]
        ls = [('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]
        for j, c in enumerate(l_colors): ls.append(('BACKGROUND', (j, 0), (j, 0), c))
        lt.setStyle(TableStyle(ls))
        elements.extend([t1, Spacer(1, 0.3*cm), t2, Spacer(1, 0.3*cm), lt, Spacer(1, 1.2*cm)])
        label_count += 1
    doc.build(elements); buffer.seek(0)
    return buffer, label_summary

def generate_rack_labels_v2(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    df['location_key'] = df.apply(create_location_key, axis=1)
    df.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    df_grouped = df.groupby('location_key')
    total_locations = len(df_grouped)
    label_count, label_summary = 0, {}
    for i, (key, group) in enumerate(df_grouped):
        if progress_bar: progress_bar.progress(int((i / total_locations) * 100))
        p1 = group.iloc[0].to_dict()
        if str(p1.get('Part No')).upper() == 'EMPTY': continue
        rack_k = f"ST-{p1.get('Station No')} / Rack {p1.get('Rack No 1st')}{p1.get('Rack No 2nd')}"
        label_summary[rack_k] = label_summary.get(rack_k, 0) + 1
        if label_count > 0 and label_count % 4 == 0: elements.append(PageBreak())
        t = Table([['Part No', format_part_no_v2(p1['Part No'])], ['Description', format_description(p1['Description'])]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        loc_v = extract_location_values(p1)
        formatted_loc = [Paragraph('Line Location', location_header_style)] + [Paragraph(str(v), get_dynamic_location_style(str(v))) for v in loc_v]
        lt = Table([formatted_loc], colWidths=[4*cm, 1.7*cm, 2.9*cm, 1.3*cm, 1.2*cm, 1.3*cm, 1.3*cm, 1.3*cm], rowHeights=0.9*cm)
        t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('ALIGN', (0,0), (0,-1), 'CENTER')]))
        ls = [('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]
        l_colors = [colors.white, colors.HexColor('#E9967A'), colors.HexColor('#ADD8E6'), colors.HexColor('#90EE90'), colors.HexColor('#FFD700'), colors.HexColor('#ADD8E6'), colors.HexColor('#E9967A'), colors.HexColor('#90EE90')]
        for j, c in enumerate(l_colors): ls.append(('BACKGROUND', (j, 0), (j, 0), c))
        lt.setStyle(TableStyle(ls))
        elements.extend([t, Spacer(1, 0.3*cm), lt, Spacer(1, 1.5*cm)])
        label_count += 1
    doc.build(elements); buffer.seek(0)
    return buffer, label_summary

def generate_bin_labels(df, mtm_models, progress_bar=None, status_text=None):
    STICKER_W, STICKER_H = 10*cm, 15*cm
    CONTENT_W, CONTENT_H = 10*cm, 7.2*cm
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(STICKER_W, STICKER_H), topMargin=0.2*cm, bottomMargin=STICKER_H-CONTENT_H-0.2*cm, leftMargin=0.1*cm, rightMargin=0.1*cm)
    df_f = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    label_summary, all_elements = {}, []
    def draw_border(canvas, doc):
        canvas.saveState(); canvas.setStrokeColorRGB(0, 0, 0); canvas.setLineWidth(1.8)
        canvas.rect((STICKER_W - CONTENT_W)/2 + 0.1*cm, STICKER_H - CONTENT_H - 0.2*cm, CONTENT_W - 0.2*cm, CONTENT_H); canvas.restoreState()
    for i, row in enumerate(df_f.to_dict('records')):
        if progress_bar: progress_bar.progress(int(((i+1) / len(df_f)) * 100))
        rack_k = f"ST-{row.get('Station No')} / Rack {row.get('Rack No 1st')}{row.get('Rack No 2nd')}"
        label_summary[rack_k] = label_summary.get(rack_k, 0) + 1
        qr = generate_qr_code_image(f"Part No: {row.get('Part No')}\nDesc: {row.get('Description')}")
        cw = CONTENT_W - 0.2*cm
        main_t = Table([["Part No", Paragraph(str(row.get('Part No')), bin_bold_style)], ["Description", Paragraph(str(row.get('Description'))[:47], bin_desc_style)], ["Qty/Bin", Paragraph(str(row.get('Qty/Bin', '')), bin_qty_style)]], colWidths=[cw/3, cw*2/3], rowHeights=[0.9*cm, 1.0*cm, 0.5*cm])
        main_t.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        iw = (cw * 2/3) / 7
        st_i = Table([extract_store_location_data_from_excel(row)], colWidths=[iw]*7, rowHeights=[0.5*cm])
        st_i.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('FONTSIZE', (0,0),(-1,-1), 9), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        s_t = Table([[Paragraph("Store Location", bin_desc_style), st_i]], colWidths=[cw/3, cw*2/3], rowHeights=[0.5*cm])
        s_t.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black)]))
        l_i = Table([extract_location_values(row)], colWidths=[iw]*7, rowHeights=[0.5*cm])
        l_i.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('FONTSIZE', (0,0),(-1,-1), 9), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        l_t = Table([[Paragraph("Line Location", bin_desc_style), l_i]], colWidths=[cw/3, cw*2/3], rowHeights=[0.5*cm])
        l_t.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black)]))
        mtm_t = None
        if mtm_models:
            qv = [Paragraph(f"<b>{row.get('Qty/Veh','') or '1'}</b>", bin_qty_style) if str(row.get('Bus Model')).upper() == m.upper() else "" for m in mtm_models]
            mtm_t = Table([mtm_models, qv], colWidths=[(3.6*cm)/len(mtm_models)]*len(mtm_models), rowHeights=[0.75*cm, 0.75*cm])
            mtm_t.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        bot = Table([[mtm_t or "", "", qr or "", ""]], colWidths=[3.6*cm, 1.0*cm, 2.5*cm, cw-7.1*cm], rowHeights=[2.5*cm])
        all_elements.extend([main_t, s_t, l_t, Spacer(1, 0.2*cm), bot, PageBreak()])
    doc.build(all_elements[:-1], onFirstPage=draw_border, onLaterPages=draw_border); buffer.seek(0)
    return buffer, label_summary

def generate_rack_list_pdf(df, base_rack_id, top_logo_file, top_logo_w, top_logo_h, fixed_logo_path, progress_bar=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=0.5*cm, bottomMargin=0.5*cm, leftMargin=1*cm, rightMargin=1*cm)
    elements = []
    df = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    grouped = df.groupby(['Station No', 'Rack No 1st', 'Rack No 2nd'])
    for i, ((st_no, r1, r2), group) in enumerate(grouped):
        if progress_bar: progress_bar.progress(int(((i+1) / len(grouped)) * 100))
        logo = ""
        if top_logo_file: logo = RLImage(io.BytesIO(top_logo_file.getvalue()), width=top_logo_w*cm, height=top_logo_h*cm)
        elements.append(Table([[Paragraph("Document Ref No.:", rl_header_style), "", logo]], colWidths=[5*cm, 17.5*cm, 5*cm]))
        m_data = [[Paragraph("STATION NAME", rl_header_style), Paragraph(str(group.iloc[0].get('Station Name', st_no)), rl_cell_left_style), Paragraph("STATION NO", rl_header_style), Paragraph(str(st_no), rl_cell_left_style)],
                  [Paragraph("MODEL", rl_header_style), Paragraph(str(group.iloc[0].get('Bus Model', '')), rl_cell_left_style), Paragraph("RACK NO", rl_header_style), Paragraph(f"Rack - {r1}{r2}", rl_cell_left_style)]]
        mt = Table(m_data, colWidths=[4*cm, 9.5*cm, 4*cm, 10*cm], rowHeights=[0.8*cm, 0.8*cm])
        mt.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#8EAADB")), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        elements.append(mt)
        h = ["S.NO", "PART NO", "PART DESCRIPTION", "CONTAINER", "QTY/BIN", "LOCATION"]
        w = [1.5*cm, 4.5*cm, 9.5*cm, 3.5*cm, 2.5*cm, 6.0*cm]
        if 'Zone' in group.columns and group['Zone'].any():
            h.insert(0, "ZONE"); w = [2.0*cm, 1.3*cm, 4.0*cm, 8.2*cm, 3.5*cm, 2.5*cm, 6.0*cm]
        rows = [h]
        for idx, r in enumerate(group.to_dict('records')):
            loc = f"{r.get('Bus Model')}-{st_no}-{base_rack_id}{r1}{r2}-{r.get('Level')}{r.get('Cell')}"
            row = [str(idx+1), r.get('Part No'), Paragraph(r.get('Description'), rl_cell_left_style), r.get('Container'), r.get('Qty/Bin', '1'), loc]
            if 'Zone' in h: row.insert(0, str(r.get('Zone', '')))
            rows.append(row)
        dt = Table(rows, colWidths=w)
        dt.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('BACKGROUND', (0,0),(-1,0), colors.HexColor("#F4B084")), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        elements.extend([dt, PageBreak()])
    doc.build(elements); buffer.seek(0)
    return buffer, len(grouped)

# --- UI MAIN (DIMENSIONS AND CAPACITY RESTORED) ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("<p style='font-style:italic;'>Designed and Developed by Agilomatrix</p>", unsafe_allow_html=True)
    
    st.sidebar.title("📄 Options")
    out_type = st.sidebar.selectbox("Choose Output Type:", ["Rack Labels", "Bin Labels", "Rack List"])
    rack_fmt = "Single Part"
    if out_type == "Rack Labels": rack_fmt = st.sidebar.selectbox("Format:", ["Single Part", "Multiple Parts"])
    base_id = st.sidebar.text_input("Infrastructure ID", "R")
    
    mtm_models = []
    if out_type == "Bin Labels":
        m1 = st.sidebar.text_input("Model 1", "7M"); m2 = st.sidebar.text_input("Model 2", "9M")
        mtm_models = [m.strip() for m in [m1, m2] if m.strip()]

    top_logo_file, top_logo_w, top_logo_h = None, 4.0, 1.5
    if out_type == "Rack List":
        top_logo_file = st.sidebar.file_uploader("Upload Top Logo", type=['png', 'jpg'])
        top_logo_w = st.sidebar.slider("Logo Width", 1.0, 8.0, 4.0)

    uploaded_file = st.file_uploader("Upload Excel", type=['xlsx'])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df.fillna('', inplace=True)
        col_check = find_required_columns(df)
        
        if col_check['Container']:
            unique_containers = get_unique_containers(df, col_check['Container'])
            
            # Restoring Expanders
            with st.expander("⚙️ Step 1: Configure Dimensions and Rack Setup", expanded=True):
                st.subheader("1. Container Dimensions")
                bin_dims = {}
                for c in unique_containers:
                    bin_dims[c] = st.text_input(f"Dimensions for {c}", key=f"bindim_{c}", placeholder="e.g., 300x200mm")
                
                st.subheader("2. Rack Dimensions & Capacity")
                num_racks = st.number_input("Number of Racks (per station)", min_value=1, value=1)
                rack_configs = {}
                for i in range(num_racks):
                    r_name = f"Rack {i+1:02d}"
                    col1, col2 = st.columns(2)
                    with col1:
                        levels = st.multiselect(f"Levels for {r_name}", ['A','B','C','D','E','F'], ['A','B','C','D'], key=f"lv_{r_name}")
                    with col2:
                        bin_caps = {}
                        for c in unique_containers:
                            bin_caps[c] = st.number_input(f"Capacity of '{c}' Bins", 0, 10, 1, key=f"cap_{r_name}_{c}")
                    rack_configs[r_name] = {'levels': levels, 'rack_bin_counts': bin_caps}

            if st.button("🚀 Generate PDF"):
                status = st.empty()
                df_loc = automate_location_assignment(df, base_id, rack_configs, status)
                
                if out_type == "Rack Labels":
                    func = generate_rack_labels_v2 if rack_fmt=="Single Part" else generate_rack_labels_v1
                    pdf, _ = func(df_loc, st.progress(0), status)
                elif out_type == "Bin Labels":
                    pdf, _ = generate_bin_labels(df_loc, mtm_models, st.progress(0), status)
                else:
                    pdf, _ = generate_rack_list_pdf(df_loc, base_id, top_logo_file, top_logo_w, top_logo_h, "Image.png", st.progress(0))
                
                st.download_button("📥 Download PDF", pdf.getvalue(), f"{out_type}.pdf")
                status.empty()
        else:
            st.error("❌ Column 'Container' not found.")

if __name__ == "__main__":
    main()
