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
bold_style_v2 = ParagraphStyle(name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=32, wordWrap='CJK')
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16)
location_header_style = ParagraphStyle(name='LocHeader', fontName='Helvetica', fontSize=16, alignment=TA_CENTER)
location_value_style_base = ParagraphStyle(name='LocValueBase', fontName='Helvetica', alignment=TA_CENTER)

bin_bold_style = ParagraphStyle(name='BinBold', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, leading=14)
bin_desc_style = ParagraphStyle(name='BinDesc', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
bin_qty_style = ParagraphStyle(name='BinQty', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)

rl_header_style = ParagraphStyle(name='RL_Header', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)
rl_cell_style = ParagraphStyle(name='RL_Cell', fontName='Helvetica', fontSize=9, alignment=TA_CENTER)
rl_cell_left_style = ParagraphStyle(name='RL_Cell_Left', fontName='Helvetica', fontSize=9, alignment=TA_LEFT)

# --- Formatting Helpers ---
def format_part_no_v1(part_no):
    if not part_no: return Paragraph("EMPTY", bold_style_v1)
    p_str = str(part_no)
    if len(p_str) > 5:
        return Paragraph(f"<b><font size=17>{p_str[:-5]}</font><font size=22>{p_str[-5:]}</font></b>", bold_style_v1)
    return Paragraph(f"<b><font size=17>{p_str}</font></b>", bold_style_v1)

def format_part_no_v2(part_no):
    p_str = str(part_no) if part_no else "EMPTY"
    if p_str.upper() == 'EMPTY': return Paragraph(f"<b><font size=34>EMPTY</font></b>", bold_style_v2)
    if len(p_str) > 5:
        return Paragraph(f"<b><font size=34>{p_str[:-5]}</font><font size=40>{p_str[-5:]}</font></b>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{p_str}</font></b>", bold_style_v2)

def format_description_v1(desc):
    d_str = str(desc) if desc else ""
    f_size = 15 if len(d_str) <= 30 else 11 if len(d_str) <= 60 else 9
    style = ParagraphStyle(name='D1', fontName='Helvetica', fontSize=f_size, leading=f_size+2)
    return Paragraph(d_str, style)

def get_dynamic_location_style(text, column_type):
    t_len = len(str(text))
    f_size = 16
    if column_type == 'Bus Model' and t_len > 4: f_size = 11
    elif column_type == 'Station No' and t_len > 3: f_size = 14
    return ParagraphStyle(name='Dyn', parent=location_value_style_base, fontSize=f_size)

def parse_dimensions(dim_str):
    if not isinstance(dim_str, str) or not dim_str: return 0, 0
    nums = [int(n) for n in re.findall(r'\d+', dim_str)]
    return (nums[0], nums[1]) if len(nums) >= 2 else (nums[0], 0) if len(nums)==1 else (0,0)

# --- DIMENSION-BASED AUTOMATION LOGIC ---

def automate_location_assignment(df, base_rack_id, rack_configs, bin_rules, status_text=None):
    # Standardize column names
    cols_map = {col.upper().strip(): col for col in df.columns}
    p_col = cols_map.get('PART NO') or cols_map.get('PART NUMBER')
    c_col = cols_map.get('CONTAINER') or cols_map.get('BIN TYPE')
    s_col = cols_map.get('STATION NO') or cols_map.get('STATION')
    d_col = cols_map.get('DESCRIPTION') or cols_map.get('DESC')
    m_col = cols_map.get('BUS MODEL') or cols_map.get('MODEL')
    sn_col = cols_map.get('STATION NAME') or cols_map.get('ST. NAME')

    if not all([p_col, c_col, s_col]):
        st.error("❌ Column mismatch. Ensure 'Part No', 'Container', and 'Station No' exist.")
        return None

    df_proc = df.copy()
    df_proc.rename(columns={p_col:'Part No', c_col:'Container', s_col:'Station No', d_col:'Description', m_col:'Bus Model', sn_col:'Station Name'}, inplace=True)
    
    # Pre-calculate Physical Rack Slots
    available_cells = []
    for rack_name, config in sorted(rack_configs.items()):
        r_nums = ''.join(filter(str.isdigit, rack_name))
        r1, r2 = (r_nums[0], r_nums[1]) if len(r_nums) > 1 else ('0', r_nums[0])
        
        # Automation: Calculate cells per shelf based on dimensions
        cells_per_shelf = int(config['rack_w'] // config['cell_w']) if config['cell_w'] > 0 else 1
        
        for level in sorted(config['levels']):
            for c_idx in range(cells_per_shelf):
                available_cells.append({
                    'Rack': base_rack_id, 'Rack No 1st': r1, 'Rack No 2nd': r2,
                    'Level': level, 'Physical_Cell': str(c_idx + 1),
                    'Cell_Width': config['cell_w']
                })

    final_list = []
    cell_idx = 0
    
    # Group by Station to reset location pointer logic
    for station_no, station_group in df_proc.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Processing Station {station_no}...")
        
        # Sort containers (larger area first)
        station_group['bin_w'] = station_group['Container'].apply(lambda x: bin_rules.get(x, (0,0))[0])
        sorted_group = station_group.sort_values(by='bin_w', ascending=False)

        for container_type, group_df in sorted_group.groupby('Container', sort=False):
            parts = group_df.to_dict('records')
            bin_w = bin_rules.get(container_type, (0,0))[0]
            
            p_ptr = 0
            while p_ptr < len(parts):
                if cell_idx >= len(available_cells):
                    st.warning(f"⚠️ Out of space at Station {station_no}")
                    break
                
                curr_loc = available_cells[cell_idx]
                # Automation: Calculate how many bins fit in this specific cell width
                bins_per_cell = int(curr_loc['Cell_Width'] // bin_w) if bin_w > 0 else 1
                
                chunk = parts[p_ptr : p_ptr + bins_per_cell]
                for p in chunk:
                    p.update(curr_loc)
                    final_list.append(p)
                
                p_ptr += bins_per_cell
                cell_idx += 1
            
            if cell_idx >= len(available_cells): break
    
    # Fill remaining gaps with EMPTY
    for i in range(cell_idx, len(available_cells)):
        empty = {k: '' for k in df_proc.columns}
        empty.update({'Part No': 'EMPTY', 'Station No': 'N/A'})
        empty.update(available_cells[i])
        final_list.append(empty)

    # Apply Sequential Cell Numbering (Resetting per level)
    return assign_sequential_ids(pd.DataFrame(final_list))

def assign_sequential_ids(df):
    df_res = []
    for (r1, r2, lvl), group in df.groupby(['Rack No 1st', 'Rack No 2nd', 'Level'], sort=False):
        parts = group[group['Part No'] != 'EMPTY'].copy()
        empties = group[group['Part No'] == 'EMPTY'].copy()
        if not parts.empty:
            parts['Cell'] = range(1, len(parts) + 1)
        if not empties.empty:
            empties['Cell'] = empties['Physical_Cell']
        df_res.append(pd.concat([parts, empties]))
    return pd.concat(df_res)

# --- PDF GENERATION ---

def generate_qr(data):
    if not QR_AVAILABLE: return None
    qr = qrcode.QRCode(box_size=10, border=2)
    qr.add_data(data); qr.make(fit=True)
    buf = BytesIO()
    qr.make_image().save(buf, format='PNG'); buf.seek(0)
    return RLImage(buf, width=2.5*cm, height=2.5*cm)

def generate_rack_labels(df, mode="v2", progress_bar=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, margin=1*cm)
    elements = []
    
    df_f = df[df['Part No'] != 'EMPTY'].copy()
    # Unique ID to group parts that belong to the same cell/label
    df_f['label_id'] = df_f['Rack No 1st'] + df_f['Rack No 2nd'] + df_f['Level'] + df_f['Cell'].astype(str)
    groups = list(df_f.groupby('label_id', sort=False))
    
    for i, (lid, group) in enumerate(groups):
        if progress_bar: progress_bar.progress(int((i+1)/len(groups)*100))
        p1 = group.iloc[0].to_dict()
        
        if mode == "v2": # Single Part Layout
            pt = Table([['Part No', format_part_no_v2(p1['Part No'])], ['Description', format_description(p1['Description'])]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        else: # Multiple Part Layout (v1)
            p2 = group.iloc[1].to_dict() if len(group)>1 else p1
            pt1 = Table([['Part No', format_part_no_v1(p1['Part No'])], ['Description', format_description_v1(p1['Description'])]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
            pt2 = Table([['Part No', format_part_no_v1(p2['Part No'])], ['Description', format_description_v1(p2['Description'])]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
            pt = Table([[pt1], [Spacer(1,0.2*cm)], [pt2]], colWidths=[15*cm])

        # Location Table
        loc_vals = [str(p1.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']]
        formatted_loc = [Paragraph('Line Location', location_header_style)] + [Paragraph(v, get_dynamic_location_style(v, 'Default')) for v in loc_vals]
        lt = Table([formatted_loc], colWidths=[4*cm] + [1.5*cm]*7, rowHeights=0.9*cm)
        
        style = TableStyle([('GRID',(0,0),(-1,-1),1,colors.black),('VALIGN',(0,0),(-1,-1),'MIDDLE')])
        pt.setStyle(style); lt.setStyle(style)
        
        elements.extend([pt, Spacer(1, 0.3*cm), lt, Spacer(1, 1.5*cm)])
        if (i+1) % 4 == 0: elements.append(PageBreak())

    doc.build(elements); buffer.seek(0); return buffer

def generate_bin_labels(df, models, progress_bar=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(10*cm, 15*cm), margin=0.2*cm)
    df_f = df[df['Part No'] != 'EMPTY'].copy()
    elements = []
    
    for i, row in enumerate(df_f.to_dict('records')):
        if progress_bar: progress_bar.progress(int((i+1)/len(df_f)*100))
        
        qr = generate_qr(f"P:{row['Part No']}\nL:{row['Level']}-{row['Cell']}")
        
        main = Table([
            ["Part No", Paragraph(str(row['Part No']), bin_bold_style)],
            ["Description", Paragraph(str(row['Description'])[:50], bin_desc_style)],
            ["Qty/Bin", Paragraph(str(row.get('Qty/Bin','1')), bin_qty_style)]
        ], colWidths=[3*cm, 6.5*cm], rowHeights=[1*cm, 1.2*cm, 0.8*cm])
        main.setStyle(TableStyle([('GRID',(0,0),(-1,-1),1,colors.black),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
        
        # MTM Matrix logic
        mtm_data = [models, [str(row.get('Qty/Veh','')) if m.upper() in str(row.get('Bus Model','')).upper() else '' for m in models]]
        mtm_t = Table(mtm_data, colWidths=[1.2*cm]*len(models))
        mtm_t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),1,colors.black),('FONTSIZE',(0,0),(-1,-1),8)]))
        
        bottom = Table([[mtm_t, qr]], colWidths=[5*cm, 4.5*cm])
        
        elements.extend([main, Spacer(1,0.5*cm), bottom, PageBreak()])
    
    doc.build(elements); buffer.seek(0); return buffer

def generate_rack_list(df, base_id, logo):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), margin=0.5*cm)
    elements = []
    df_f = df[df['Part No'] != 'EMPTY'].copy()
    
    for (st_no, rk), group in df_f.groupby(['Station No', 'Rack No 1st'], sort=False):
        header = Table([[logo if logo else "", Paragraph(f"STATION: {st_no} | RACK: {rk}", rl_header_style)]], colWidths=[5*cm, 20*cm])
        data = [["S.NO", "PART NO", "DESCRIPTION", "CONTAINER", "LOCATION"]]
        for idx, r in enumerate(group.to_dict('records')):
            loc = f"{r['Bus Model']}-{r['Station No']}-{base_id}{r['Rack No 1st']}{r['Rack No 2nd']}-{r['Level']}{r['Cell']}"
            data.append([idx+1, r['Part No'], Paragraph(r['Description'], rl_cell_left_style), r['Container'], loc])
        
        t = Table(data, colWidths=[1.5*cm, 4*cm, 10*cm, 3*cm, 6*cm])
        t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),1,colors.black),('BACKGROUND',(0,0),(-1,0),colors.lightgrey)]))
        elements.extend([header, Spacer(1,0.2*cm), t, PageBreak()])
        
    doc.build(elements); buffer.seek(0); return buffer

# --- MAIN APP ---

def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("Designed by **Agilomatrix**")

    mode = st.sidebar.selectbox("Output Type", ["Rack Labels", "Bin Labels", "Rack List"])
    base_id = st.sidebar.text_input("Infrastructure ID", "R")
    
    file = st.file_uploader("Upload Excel", type=['xlsx'])
    if file:
        df = pd.read_excel(file).fillna('')
        
        with st.expander("⚙️ Automation & Dimensions Setup", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Rack Geometry")
                rw = st.number_input("Rack Internal Width (mm)", 100, 5000, 2400)
                cw = st.number_input("Cell (Slot) Width (mm)", 100, 2000, 600)
                nr = st.number_input("Number of Racks", 1, 50, 2)
                lvls = st.multiselect("Levels", ['A','B','C','D','E','F'], ['A','B','C','D'])
            
            with c2:
                st.subheader("Bin Rules")
                bins = df.iloc[:, df.columns.str.contains('CONTAINER', case=False)].iloc[:,0].unique() if any(df.columns.str.contains('CONTAINER', case=False)) else []
                bin_rules = {}
                for b in bins:
                    b_dim = st.text_input(f"WxD for {b}", "300x400", key=b)
                    bin_rules[b] = parse_dimensions(b_dim)

        if st.button("🚀 Generate PDF"):
            rack_configs = {f"Rack {i+1:02d}": {'rack_w':rw, 'cell_w':cw, 'levels':lvls} for i in range(nr)}
            
            status = st.empty()
            df_final = automate_location_assignment(df, base_id, rack_configs, bin_rules, status)
            
            if df_final is not None:
                progress = st.progress(0)
                if mode == "Rack Labels":
                    sub = st.sidebar.radio("Layout", ["Single Part", "Multiple Parts"])
                    pdf = generate_rack_labels(df_final, "v2" if sub=="Single Part" else "v1", progress)
                elif mode == "Bin Labels":
                    models = st.sidebar.text_input("Models (comma separated)", "7M, 9M, 12M").split(',')
                    pdf = generate_bin_labels(df_final, [m.strip() for m in models], progress)
                else:
                    pdf = generate_rack_list(df_final, base_id, None)

                st.download_button("📥 Download PDF", pdf, "SmartTag_Output.pdf")
                st.dataframe(df_final[['Part No', 'Station No', 'Level', 'Cell', 'Physical_Cell']])

if __name__ == "__main__":
    main()
