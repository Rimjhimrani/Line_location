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
    layout="wide"
)

# --- Style Definitions ---
bold_style_v2 = ParagraphStyle(
    name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=32, spaceBefore=0, spaceAfter=2, wordWrap='CJK'
)
desc_style = ParagraphStyle(
    name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16, spaceBefore=2, spaceAfter=2
)
bin_bold_style = ParagraphStyle(
    name='BinBold', fontName='Helvetica-Bold', fontSize=18, alignment=TA_CENTER, leading=20
)
bin_desc_style = ParagraphStyle(
    name='BinDesc', fontName='Helvetica', fontSize=10, alignment=TA_CENTER, leading=12
)
bin_qty_style = ParagraphStyle(
    name='BinQty', fontName='Helvetica-Bold', fontSize=14, alignment=TA_CENTER, leading=16
)
rl_header_style = ParagraphStyle(
    name='RLHeader', fontName='Helvetica-Bold', fontSize=11, alignment=TA_LEFT
)
rl_cell_left_style = ParagraphStyle(
    name='RLCellLeft', fontName='Helvetica', fontSize=10, alignment=TA_LEFT
)

# --- Formatting Functions ---
def format_part_no_v2(part_no):
    part_no = str(part_no)
    if part_no.upper() == 'EMPTY':
         return Paragraph(f"<b><font size=34>EMPTY</font></b>", bold_style_v2)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=34>{part1}</font><font size=40>{part2}</font></b>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b>", bold_style_v2)

# --- Core Logic: Column Finding & Mapping ---
def find_required_columns(df):
    cols = {col.upper().strip(): col for col in df.columns}
    part_no_key = next((k for k in cols if 'PART' in k and ('NO' in k or 'NUM' in k)), None)
    desc_key = next((k for k in cols if 'DESC' in k), None)
    bus_model_key = next((k for k in cols if 'BUS' in k and 'MODEL' in k), None)
    station_no_key = next((k for k in cols if 'STATION' in k and ('NO' in k or 'NUM' in k)), None)
    if not station_no_key: station_no_key = next((k for k in cols if 'STATION' in k), None)
    container_type_key = next((k for k in cols if 'CONTAINER' in k or 'BINTYPE' in k or 'BIN TYPE' in k), None)
    return {
        'Part No': cols.get(part_no_key), 
        'Description': cols.get(desc_key), 
        'Bus Model': cols.get(bus_model_key), 
        'Station No': cols.get(station_no_key), 
        'BinType': cols.get(container_type_key)
    }

def get_unique_bintypes(df, bintype_col):
    if not bintype_col or bintype_col not in df.columns: return []
    return sorted(df[bintype_col].dropna().astype(str).unique())

# --- Core Logic: Multi-Rack Transition Allocation ---
def generate_multi_rack_allocation(df, base_rack_id, rack_templates, bintype_dims, status_text=None):
    req = find_required_columns(df)
    df_p = df.copy()
    df_p.rename(columns={req['Part No']: 'Part No', req['Description']: 'Description', 
                         req['Bus Model']: 'Bus Model', req['Station No']: 'Station No', 
                         req['BinType']: 'BinType'}, inplace=True)
    
    final_assigned_data = []

    for station_no, station_group in df_p.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Allocating Station: {station_no}...")
        station_remaining_parts = station_group.copy()

        # Iterate through Rack Types sequentially (Type A -> Type B ...)
        for template_name, config in rack_templates.items():
            if station_remaining_parts.empty: break
                
            levels = config['levels']
            capacities = config['capacities']
            dim_val = config.get('dims', 'N/A')
            header_name = f"{template_name} ({dim_val})"
            
            curr_rack_num, curr_lvl_idx, curr_cell_idx = 1, 0, 1

            # Only pick parts where BinType capacity > 0 for THIS rack type
            allowed_bts = [bt for bt, cap in capacities.items() if cap > 0]
            parts_fitting_this_rack = station_remaining_parts[station_remaining_parts['BinType'].isin(allowed_bts)].copy()

            if parts_fitting_this_rack.empty: continue
            assigned_this_pass = []

            for bt_name, parts_subgroup in parts_fitting_this_rack.groupby('BinType', sort=True):
                bins_per_level = capacities.get(bt_name, 1)
                bt_dim_val = bintype_dims.get(bt_name, "N/A")
                
                for part in parts_subgroup.to_dict('records'):
                    if curr_cell_idx > bins_per_level:
                        curr_cell_idx, curr_lvl_idx = 1, curr_lvl_idx + 1
                    if curr_lvl_idx >= len(levels):
                        curr_lvl_idx, curr_rack_num, curr_cell_idx = 0, curr_rack_num + 1, 1

                    rack_str = f"{curr_rack_num:02d}"
                    part.update({
                        'Rack': base_rack_id, 'Rack No 1st': rack_str[0], 'Rack No 2nd': rack_str[1],
                        'Level': levels[curr_lvl_idx], 'Physical_Cell': f"{curr_cell_idx:02d}",
                        'Station No': station_no, 'Rack Key': rack_str,
                        'Rack Type Summary Header': header_name,
                        'Bin Dimension': bt_dim_val
                    })
                    final_assigned_data.append(part)
                    assigned_this_pass.append(part['Part No'])
                    curr_cell_idx += 1
            
            # Remove assigned parts before checking the next rack type for this station
            station_remaining_parts = station_remaining_parts[~station_remaining_parts['Part No'].isin(assigned_this_pass)]

    return pd.DataFrame(final_assigned_data)

def assign_sequential_ids(df):
    if df.empty: return df
    df_sorted = df.sort_values(by=['Station No', 'Rack Type Summary Header', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']).copy()
    loc_counters = {}
    sequential_ids = []
    for _, row in df_sorted.iterrows():
        key = (row['Station No'], row['Rack Type Summary Header'], row['Rack No 1st'], row['Rack No 2nd'], row['Level'])
        loc_counters[key] = loc_counters.get(key, 0) + 1
        sequential_ids.append(loc_counters[key])
    df_sorted['Cell'] = sequential_ids
    return df_sorted

# --- Core Logic: Summary Logic ---
def generate_summary_table(df):
    if df.empty: return pd.DataFrame()
    summary = df.groupby(['Station No', 'Rack Type Summary Header'])['Rack Key'].nunique().reset_index()
    pivot_df = summary.pivot(index='Station No', columns='Rack Type Summary Header', values='Rack Key').fillna(0).astype(int)
    pivot_df.loc['TOTAL'] = pivot_df.sum()
    return pivot_df

# --- PDF Helpers ---
def extract_store_data(row):
    def get_val(names):
        for n in names:
            v = row.get(n)
            if pd.notna(v) and str(v).strip().lower() not in ['nan','none','']: return str(v).strip()
        return ""
    # Fields: Store Location, ST NAME Short, Zone, Location, Floor, Rack No, Level
    return [get_val(['Store Location', 'STORELOCATION']), 
            get_val(['ST. NAME (Short)', 'ST.NAME (Short)', 'Station Name Short']), 
            get_val(['Zone', 'ABB ZONE']), get_val(['Location', 'ABB LOCATION']), 
            get_val(['Floor', 'ABB FLOOR']), get_val(['Rack No', 'ABB RACK NO']), 
            get_val(['Level', 'ABB LEVEL IN RACK'])]

def generate_qr(data):
    if not QR_AVAILABLE: return None
    qr = qrcode.QRCode(version=1, box_size=10, border=2)
    qr.add_data(data); qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
    return RLImage(buf, width=2.5*cm, height=2.5*cm)

# --- PDF Generation: Rack Labels ---
def generate_rack_labels_pdf(df, prog=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    total = len(df)
    for i, row in enumerate(df.to_dict('records')):
        if prog: prog.progress(int((i / total) * 100))
        if i > 0 and i % 4 == 0: elements.append(PageBreak())
        
        t1 = Table([['Part No', format_part_no_v2(row['Part No'])], 
                    ['Description', Paragraph(str(row['Description']), desc_style)]], 
                   colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        t1.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('ALIGN', (0,0),(0,-1), 'CENTER')]))
        
        loc_vals = ['Line Location', str(row['Bus Model']), str(row['Station No']), str(row['Rack']), str(row['Rack No 1st']), str(row['Rack No 2nd']), str(row['Level']), str(row['Cell'])]
        t2 = Table([loc_vals], colWidths=[4*cm] + [(11*cm)/7]*7, rowHeights=1.2*cm)
        t2_style = [('GRID', (0,0),(-1,-1), 1, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTSIZE', (0,0),(-1,-1), 14)]
        c_list = [colors.white, colors.HexColor('#E9967A'), colors.HexColor('#ADD8E6'), colors.HexColor('#90EE90'), colors.HexColor('#FFD700'), colors.HexColor('#ADD8E6'), colors.HexColor('#E9967A'), colors.HexColor('#90EE90')]
        for j, c in enumerate(c_list): t2_style.append(('BACKGROUND', (j,0), (j,0), c))
        t2.setStyle(TableStyle(t2_style))
        elements.extend([t1, Spacer(1, 0.3*cm), t2, Spacer(1, 0.2*cm)])
    doc.build(elements); buffer.seek(0)
    return buffer

# --- PDF Generation: Bin Labels (Stickers) ---
def generate_bin_labels_pdf(df, mtm_models, prog=None):
    STICKER_W, STICKER_H = 10*cm, 15*cm
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(STICKER_W, STICKER_H), topMargin=0.2*cm, bottomMargin=0.2*cm, leftMargin=0.1*cm, rightMargin=0.1*cm)
    elements = []
    total = len(df)
    for i, row in enumerate(df.to_dict('records')):
        if prog: prog.progress(int((i / total) * 100))
        st_data = extract_store_data(row)
        line_data = [str(row.get(c,'')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']]
        qr_img = generate_qr(f"Part:{row['Part No']}\nStore:{'|'.join(st_data)}\nLine:{'|'.join(line_data)}")
        cw = 9.8*cm
        
        # Header: Station Short Name
        h1 = Table([[Paragraph("STATION NAME (SHORT)", bin_desc_style), Paragraph(st_data[1], bin_bold_style)]], colWidths=[cw/3, 2*cw/3], rowHeights=[0.8*cm])
        h1.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('BACKGROUND', (0,0),(0,0), colors.lightgrey), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        
        # Dimension integrated with Description
        dim_info = f"<br/><font color='blue' size='8'>DIMENSION: {row.get('Bin Dimension', 'N/A')}</font>"
        m_t = Table([["Part No", Paragraph(str(row['Part No']), bin_bold_style)], 
                     ["Description", Paragraph(str(row['Description'])[:45] + dim_info, bin_desc_style)], 
                     ["Qty/Bin", Paragraph(str(row.get('Qty/Bin','')), bin_qty_style)]], 
                    colWidths=[cw/3, 2*cw/3], rowHeights=[0.9*cm, 1.2*cm, 0.5*cm])
        m_t.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        
        def sub_row(label, vls):
            iw = (2*cw/3)/7
            it = Table([vls], colWidths=[iw]*7, rowHeights=[0.5*cm])
            it.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('FONTSIZE', (0,0),(-1,-1), 8), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
            return Table([[Paragraph(label, bin_desc_style), it]], colWidths=[cw/3, 2*cw/3])
        
        mtm_t = None
        if mtm_models:
            qty_v = [Paragraph(f"<b>{row.get('Qty/Veh','')}</b>", bin_qty_style) if str(row.get('Bus Model','')).upper() == m.upper() else "" for m in mtm_models]
            mtm_t = Table([mtm_models, qty_v], colWidths=[(3.6*cm)/len(mtm_models)]*len(mtm_models), rowHeights=[0.75*cm, 0.75*cm])
            mtm_t.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))

        b_row = Table([[mtm_t or "", "", qr_img or "", ""]], colWidths=[3.6*cm, 1.0*cm, 2.5*cm, cw-7.1*cm], rowHeights=[2.5*cm])
        elements.extend([h1, m_t, sub_row("Store Location", st_data), sub_row("Line Location", line_data), Spacer(1, 0.2*cm), b_row, PageBreak()])
    doc.build(elements[:-1]); buffer.seek(0)
    return buffer

# --- PDF Generation: Rack List (Detailed Table) ---
def generate_rack_list_pdf(df, base_rack_id):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=0.5*cm, bottomMargin=0.5*cm, leftMargin=1*cm, rightMargin=1*cm)
    elements = []
    grouped = df.groupby(['Station No', 'Rack Key'])
    for (st_no, r_key), group in grouped:
        first = group.iloc[0]
        st_name = extract_store_data(first)[1]
        m_data = [[Paragraph("STATION NAME", rl_header_style), Paragraph(st_name, rl_cell_left_style), 
                   Paragraph("STATION NO", rl_header_style), Paragraph(str(st_no), rl_cell_left_style)],
                  [Paragraph("MODEL", rl_header_style), Paragraph(str(first.get('Bus Model','')), rl_cell_left_style), 
                   Paragraph("RACK NO", rl_header_style), Paragraph(f"Rack - {r_key}", rl_cell_left_style)]]
        mt = Table(m_data, colWidths=[4*cm, 9.5*cm, 4*cm, 10*cm], rowHeights=[0.8*cm]*2)
        mt.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('BACKGROUND', (0,0),(-1,-1), colors.HexColor("#8EAADB")), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        
        # Detailed Columns
        data = [["S.NO", "PART NO", "PART DESCRIPTION", "BINTYPE", "DIMENSION", "QTY/BIN", "LOCATION"]]
        for idx, r in enumerate(group.to_dict('records')):
            loc = f"{r.get('Bus Model')}-{r.get('Station No')}-{base_rack_id}{r_key}-{r.get('Level')}{r.get('Cell')}"
            data.append([idx+1, r['Part No'], Paragraph(str(r['Description']), rl_cell_left_style), 
                         r['BinType'], r.get('Bin Dimension', 'N/A'), r['Qty/Bin'], loc])
        
        t = Table(data, colWidths=[1.2*cm, 3.8*cm, 8.5*cm, 3.2*cm, 3.0*cm, 2.0*cm, 6.0*cm])
        t.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F4B084")), 
                               ('ALIGN', (0,0), (-1, -1), 'CENTER'), ('VALIGN', (0,0), (-1, -1), 'MIDDLE'), ('FONTSIZE', (0,0), (-1, -1), 9)]))
        elements.extend([mt, Spacer(1, 0.2*cm), t, PageBreak()])
    doc.build(elements[:-1]); buffer.seek(0)
    return buffer

# --- Main Application Interface ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("<p style='font-style:italic;'>Designed by Rimjhim Rani | Agilomatrix</p>", unsafe_allow_html=True)
    
    st.sidebar.title("📄 Config")
    output_type = st.sidebar.selectbox("Choose Output Type:", ["Rack Labels", "Bin Labels", "Rack List"])
    
    m1, m2, m3 = "7M", "9M", "12M"
    if output_type == "Bin Labels":
        m1, m2, m3 = st.sidebar.text_input("Model 1", "7M"), st.sidebar.text_input("Model 2", "9M"), st.sidebar.text_input("Model 3", "12M")

    base_rack_id = st.sidebar.text_input("Infrastructure ID", "R")
    uploaded_file = st.file_uploader("Upload Data (Excel/CSV)", type=['xlsx', 'xls', 'csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"✅ Loaded {len(df)} parts.")
        req = find_required_columns(df)
        
        if req['BinType'] and req['Station No']:
            st.sidebar.markdown("---")
            num_types = st.sidebar.number_input("Number of Rack Types", 1, 5, 1)
            unique_bts = get_unique_bintypes(df, req['BinType'])
            
            # --- RACK TYPE CONFIG ---
            rack_templates = {}
            for i in range(int(num_types)):
                st.sidebar.subheader(f"Rack Type {i+1}")
                r_n, r_d = st.sidebar.text_input("Rack Name", f"Type {chr(65+i)}", key=f"n_{i}"), st.sidebar.text_input("Dimensions", "2400*800", key=f"d_{i}")
                r_l = st.sidebar.multiselect("Levels", ['A','B','C','D','E','F'], default=['A','B','C','D'], key=f"l_{i}")
                st.sidebar.caption("Shelf Capacity (0 = skip in this rack type)")
                caps = {bt: st.sidebar.number_input(f"{bt} cap", 0, 50, 4, key=f"c_{i}_{bt}") for bt in unique_bts}
                rack_templates[r_n] = {'levels': r_l, 'capacities': caps, 'dims': r_d}

            # --- BINTYPE DIMENSION CONFIG ---
            st.sidebar.markdown("---")
            st.sidebar.subheader("BinType Dimensions")
            bt_dims = {bt: st.sidebar.text_input(f"{bt} Dimension (L x W)", "600x400", key=f"bt_dim_{bt}") for bt in unique_bts}

            if st.button("🚀 Process & Generate Output", type="primary"):
                status = st.empty()
                # 1. Multi-Allocation Logic
                df_alloc = generate_multi_rack_allocation(df, base_rack_id, rack_templates, bt_dims, status)
                df_final = assign_sequential_ids(df_alloc)
                
                if not df_final.empty:
                    # 2. Utilization Summary Table
                    st.subheader("📊 Station-Wise Rack Summary")
                    summary = generate_summary_table(df_final)
                    st.table(summary.style.format(precision=0).highlight_max(axis=0, color='#e6f3ff'))
                    
                    # 3. File Downloads
                    st.markdown("---")
                    st.subheader("📥 Downloads")
                    col1, col2 = st.columns(2)
                    ex_buf = io.BytesIO()
                    df_final.to_excel(ex_buf, index=False)
                    col1.download_button("📥 Download Excel Allocation", ex_buf.getvalue(), "Allocation.xlsx")
                    
                    prog = st.progress(0)
                    if output_type == "Rack Labels":
                        pdf = generate_rack_labels_pdf(df_final, prog)
                        col2.download_button("📥 Download Rack Labels PDF", pdf, "Rack_Labels.pdf")
                    elif output_type == "Bin Labels":
                        mtm = [m.strip() for m in [m1, m2, m3] if m.strip()]
                        pdf = generate_bin_labels_pdf(df_final, mtm, prog)
                        col2.download_button("📥 Download Bin Labels PDF", pdf, "Bin_Labels.pdf")
                    elif output_type == "Rack List":
                        pdf = generate_rack_list_pdf(df_final, base_rack_id)
                        col2.download_button("📥 Download Rack List PDF", pdf, "Rack_List.pdf")
                    prog.empty(); status.success("All Steps Completed Successfully!")
                else:
                    st.warning("No parts allocated. Please check that capacities are not all set to zero.")
        else:
            st.error("Missing columns: Ensure 'Station No' and 'BinType/Container' exist in file.")

if __name__ == "__main__":
    main()
