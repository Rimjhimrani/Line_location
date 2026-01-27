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

# --- Style Definitions for Rack Labels ---
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

# --- Style Definitions for Bin Labels ---
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

# --- Style Definitions for Rack List ---
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

# --- Formatting Functions ---
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

# --- Core Logic Functions ---
def find_required_columns(df):
    cols = {col.upper().strip(): col for col in df.columns}
    part_no_key = next((k for k in cols if 'PART' in k and ('NO' in k or 'NUM' in k)), None)
    desc_key = next((k for k in cols if 'DESC' in k), None)
    bus_model_key = next((k for k in cols if 'BUS' in k and 'MODEL' in k), None)
    station_no_key = next((k for k in cols if 'STATION' in k and 'NO' in k), None)
    if not station_no_key:
        station_no_key = next((k for k in cols if 'STATION' in k), None)
    container_type_key = next((k for k in cols if 'CONTAINER' in k), None)
    
    # Specific keys for different Station Names
    st_name_full_key = next((k for k in df.columns if k.upper().strip() == 'STATION NAME'), None)
    st_name_short_key = next((k for k in df.columns if 'SHORT' in k.upper() or 'ST. NAME' in k.upper()), None)

    return {
        'Part No': cols.get(part_no_key),
        'Description': cols.get(desc_key),
        'Bus Model': cols.get(bus_model_key),
        'Station No': cols.get(station_no_key),
        'Container': cols.get(container_type_key),
        'ST_Full': st_name_full_key,
        'ST_Short': st_name_short_key
    }

def get_unique_containers(df, container_col):
    if not container_col or container_col not in df.columns: return []
    return sorted(df[container_col].dropna().astype(str).unique())

def generate_station_wise_assignment(df, base_rack_id, levels, cells_per_level, bin_info_map, status_text=None):
    req = find_required_columns(df)
    df_processed = df.copy()
    
    # Rename for internal logic while keeping original columns for metadata
    rename_dict = {
        req['Part No']: 'Part No', req['Description']: 'Description',
        req['Bus Model']: 'Bus Model', req['Station No']: 'Station No', req['Container']: 'Container'
    }
    # Track extra station columns if they exist
    if req['ST_Full']: rename_dict[req['ST_Full']] = 'ST_Full_Internal'
    if req['ST_Short']: rename_dict[req['ST_Short']] = 'ST_Short_Internal'
    
    df_processed.rename(columns={k: v for k, v in rename_dict.items() if k}, inplace=True)
    df_processed['bin_info'] = df_processed['Container'].map(bin_info_map)
    df_processed['bins_per_cell'] = df_processed['bin_info'].apply(lambda x: x['capacity'] if x else 1)
    
    final_assigned_data = []

    for station_no, station_group in df_processed.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Processing Station: {station_no}...")
        
        station_cells_needed = 0
        container_groups = sorted(station_group.groupby('Container'), key=lambda x: x[0], reverse=True)
        
        for _, cont_df in container_groups:
            cap = cont_df['bins_per_cell'].iloc[0]
            station_cells_needed += math.ceil(len(cont_df) / cap)
        
        cells_per_rack = len(levels) * cells_per_level
        racks_for_station = math.ceil(station_cells_needed / cells_per_rack)
        
        station_available_cells = []
        for r_idx in range(1, racks_for_station + 1):
            rack_str = f"{r_idx:02d}"
            r1, r2 = rack_str[0], rack_str[1]
            for lvl in sorted(levels):
                for c_idx in range(1, cells_per_level + 1):
                    station_available_cells.append({
                        'Rack No 1st': r1, 'Rack No 2nd': r2, 
                        'Level': lvl, 'Physical_Cell': f"{c_idx:02d}", 
                        'Rack': base_rack_id
                    })

        current_cell_ptr = 0
        for _, cont_df in container_groups:
            parts = cont_df.to_dict('records')
            cap = parts[0]['bins_per_cell']
            
            for i in range(0, len(parts), cap):
                chunk = parts[i:i + cap]
                if current_cell_ptr < len(station_available_cells):
                    loc = station_available_cells[current_cell_ptr]
                    for p in chunk:
                        p.update(loc)
                        final_assigned_data.append(p)
                    current_cell_ptr += 1
        
        for i in range(current_cell_ptr, len(station_available_cells)):
            empty_label = {
                'Part No': 'EMPTY', 'Description': '', 'Bus Model': station_group['Bus Model'].iloc[0] if not station_group.empty else '', 
                'Station No': station_no, 'Container': '',
                'ST_Full_Internal': station_group['ST_Full_Internal'].iloc[0] if 'ST_Full_Internal' in station_group.columns else '',
                'ST_Short_Internal': station_group['ST_Short_Internal'].iloc[0] if 'ST_Short_Internal' in station_group.columns else ''
            }
            empty_label.update(station_available_cells[i])
            final_assigned_data.append(empty_label)

    return pd.DataFrame(final_assigned_data)

def generate_by_rack_type(df, base_rack_id, rack_templates, container_configs, status_text=None):
    req = find_required_columns(df)
    df_p = df.copy()
    rename_dict = {
        req['Part No']: 'Part No', req['Description']: 'Description', 
        req['Bus Model']: 'Bus Model', req['Station No']: 'Station No', req['Container']: 'Container'
    }
    if req['ST_Full']: rename_dict[req['ST_Full']] = 'ST_Full_Internal'
    if req['ST_Short']: rename_dict[req['ST_Short']] = 'ST_Short_Internal'
    
    df_p.rename(columns={k: v for k, v in rename_dict.items() if k}, inplace=True)
    
    final_data = []
    if not rack_templates: return pd.DataFrame()
    template_name = list(rack_templates.keys())[0]
    config = rack_templates[template_name]
    levels = config['levels']

    for station_no, station_group in df_p.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Allocating Station: {station_no}...")
        curr_rack_num = 1
        curr_lvl_idx = 0
        curr_cell_idx = 1
        
        for cont_type, parts_group in station_group.groupby('Container', sort=True):
            bins_per_level = config['capacities'].get(cont_type, 1)
            all_parts = parts_group.to_dict('records')

            for part in all_parts:
                if curr_cell_idx > bins_per_level:
                    curr_cell_idx = 1
                    curr_lvl_idx += 1
                
                if curr_lvl_idx >= len(levels):
                    curr_lvl_idx = 0
                    curr_rack_num += 1
                    curr_cell_idx = 1

                rack_str = f"{curr_rack_num:02d}"
                part.update({
                    'Rack': base_rack_id, 'Rack No 1st': rack_str[0], 'Rack No 2nd': rack_str[1],
                    'Level': levels[curr_lvl_idx], 'Physical_Cell': f"{curr_cell_idx:02d}",
                    'Station No': station_no, 'Rack Key': rack_str
                })
                final_data.append(part)
                curr_cell_idx += 1
    return pd.DataFrame(final_data)
    
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
    
    # Check internal column first
    station_name_short = str(row.get('ST_Short_Internal', get_clean_value(['ST. NAME (Short)', 'ST.NAME (Short)', 'ST NAME (Short)', 'Station Name Short'])))
    
    # REQUIRED ORDER: [Store Location, Station Name (Short), Zone, Location, Floor, Rack No, Level]
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

# --- PDF Generation: Rack Labels ---
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

# --- PDF Generation: Bin Labels ---
def generate_bin_labels(df, mtm_models, progress_bar=None, status_text=None):
    if not QR_AVAILABLE:
        st.error("❌ QR Code library not found.")
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
        if status_text: status_text.text(f"Processing Bin Label {i+1}/{total_labels}")
        
        # Bin Label header uses ST. NAME (Short)
        st_short = str(row.get('ST_Short_Internal', 'NA'))
        rack_key = f"{st_short} / Rack {row.get('Rack No 1st', '0')}{row.get('Rack No 2nd', '0')}"
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
        
        # BAR ORDER: [Store Loc, Station Short, Zone, Location, Floor, Rack No, Level]
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
            bus_model = str(row.get('Bus Model', '')).strip().upper()
            mtm_qty_values = [Paragraph(f"<b>{qty_veh}</b>", bin_qty_style) if bus_model == model.strip().upper() and qty_veh else "" for model in mtm_models]
            mtm_data = [mtm_models, mtm_qty_values]
            num_models = len(mtm_models)
            col_width = (3.6 * cm) / num_models if num_models > 0 else 3.6 * cm
            mtm_table = Table(mtm_data, colWidths=[col_width] * num_models, rowHeights=[0.75*cm, 0.75*cm])
            mtm_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTNAME', (0,0),(-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0),(-1,-1), 9)]))

        bottom_row = Table([[mtm_table or "", "", qr_image or "", ""]], colWidths=[3.6 * cm, 1.0 * cm, 2.5 * cm, content_width - 7.1*cm], rowHeights=[2.5*cm])
        bottom_row.setStyle(TableStyle([('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))

        all_elements.extend([main_table, store_loc_table, line_loc_table, Spacer(1, 0.2*cm), bottom_row])
        if i < total_labels - 1:
            all_elements.append(PageBreak())

    if all_elements: doc.build(all_elements, onFirstPage=draw_border, onLaterPages=draw_border)
    buffer.seek(0)
    return buffer, label_summary

# --- PDF Generation: Rack List (Uses Full STATION NAME) ---
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
        if status_text: status_text.text(f"Generating List for Station {station_no} / Rack {rack_key}")
        
        first_row = group.iloc[0]
        # Rack list header uses Full STATION NAME
        st_full = str(first_row.get('ST_Full_Internal', 'NA'))
        bus_model = str(first_row.get('Bus Model', ''))
        
        top_logo_img = ""
        if top_logo_file:
            try:
                img_io = io.BytesIO(top_logo_file.getvalue())
                top_logo_img = RLImage(img_io, width=top_logo_w*cm, height=top_logo_h*cm)
            except:
                pass
        
        header_table = Table([[Paragraph("Document Ref No.:", rl_header_style), "", top_logo_img]], colWidths=[5*cm, 17.5*cm, 5*cm])
        header_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP'), ('ALIGN', (-1,-1), (-1,-1), 'RIGHT')]))
        elements.append(header_table)
        elements.append(Spacer(1, 0.1*cm))
        
        master_data = [
            [Paragraph("STATION NAME", ParagraphStyle('H', fontName='Helvetica-Bold', fontSize=12)), 
             Paragraph(st_full, master_value_style_left),
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
            ('GRID', (0,0), (-1, -1), 1, colors.black),
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
        
        for idx, row in enumerate(group_sorted.to_dict('records')):
            loc_str = f"{bus_model}-{row.get('Station No','')}-{base_rack_id}{rack_key}-{row.get('Level','')}{row.get('Cell','')}"
            row_data = [str(idx + 1), str(row.get('Part No', '')), Paragraph(str(row.get('Description', '')), rl_cell_left_style), str(row.get('Container', '')), str(row.get('Qty/Bin', '')), loc_str]
            if has_zone: row_data.insert(0, str(row.get('Zone', '')))
            data_rows.append(row_data)
            
        data_table = Table(data_rows, colWidths=col_widths)
        data_table.setStyle(TableStyle(table_style_cmds))
        elements.append(data_table)
        elements.append(Spacer(1, 0.2*cm))
        
        today_date = datetime.date.today().strftime("%d-%m-%Y")
        fixed_logo_path = "Image.png"
        fixed_logo_img = Paragraph("<b>[Agilomatrix Logo Missing]</b>", rl_cell_left_style)
        if os.path.exists(fixed_logo_path):
            try: fixed_logo_img = RLImage(fixed_logo_path, width=4.3*cm, height=1.5*cm)
            except: pass
        
        left_content = [
            Paragraph(f"<i>Creation Date: {today_date}</i>", rl_cell_left_style),
            Spacer(1, 0.2*cm),
            Paragraph("<b>Verified by:</b>", rl_header_style),
            Paragraph("Name: ___________________", rl_cell_left_style),
            Paragraph("Signature: _______________", rl_cell_left_style)
        ]
        designed_by_text = Paragraph("Designed by:", rl_header_style)
        right_inner_table = Table([[designed_by_text, fixed_logo_img]], colWidths=[3*cm, 4.5*cm])
        right_inner_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('ALIGN', (0,0), (-1,-1), 'RIGHT')]))

        footer_table = Table([[left_content, right_inner_table]], colWidths=[20*cm, 7.7*cm])
        footer_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'BOTTOM')]))
        
        elements.append(footer_table)
        elements.append(PageBreak())
        
    doc.build(elements)
    buffer.seek(0)
    return buffer, total_groups

# --- Main Application UI ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("<p style='font-style:italic;'>Designed and Developed by Rimjhim Rani | Agilomatrix</p>", unsafe_allow_html=True)
    
    st.sidebar.title("📄 Config")
    output_type = st.sidebar.selectbox("Choose Output Type:", ["Rack Labels", "Bin Labels", "Rack List"])
    
    # Vehicle Models for MTM Table
    model1, model2, model3 = "7M", "9M", "12M"
    if output_type == "Bin Labels":
        model1 = st.sidebar.text_input("Model 1", "7M")
        model2 = st.sidebar.text_input("Model 2", "9M")
        model3 = st.sidebar.text_input("Model 3", "12M")

    base_rack_id = st.sidebar.text_input("Infrastructure ID", "R")
    generation_method = st.sidebar.radio("Generation Method:", ["By Cell Dimension", "By Rack Type"])
    
    uploaded_file = st.file_uploader("Upload Your Data Here (Excel/CSV)", type=['xlsx', 'xls', 'csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"✅ Loaded {len(df)} parts.")
        req_cols = find_required_columns(df)
        
        if req_cols['Container'] and req_cols['Station No']:
            st.sidebar.markdown("---")
            
            if generation_method == "By Cell Dimension":
                st.sidebar.subheader("Global Rack Settings")
                # Common Fields
                cell_dim_common = st.sidebar.text_input("Cell Dimension (L x W)", "600x400")
                levels = st.sidebar.multiselect("Active Levels", options=['A','B','C','D','E','F','G','H'], default=['A','B','C','D'])
                num_cells = st.sidebar.number_input("Cells per Level", min_value=1, value=10)
                
                unique_c = get_unique_containers(df, req_cols['Container'])
                bin_rules = {}
                # Specific Fields per Container
                for c in unique_c:
                    st.sidebar.markdown(f"#### Container: {c}")
                    b_dim = st.sidebar.text_input(f"BinType / Container Dimension - {c}", "400x300", key=f"bd_{c}")
                    cap = st.sidebar.number_input(f"capacity of container per cell - {c}", min_value=1, value=1, key=f"c_{c}")
                    bin_rules[c] = {'capacity': cap, 'bin_dim': b_dim, 'cell_dim': cell_dim_common}
            
            else:  # --- By Rack Type ---
                st.sidebar.subheader("1. Rack Type Configuration")
                num_rack_types = st.sidebar.number_input("Number of Rack Types", 1, 5, 1)
                unique_c = get_unique_containers(df, req_cols['Container'])
                rack_templates = {}
                for i in range(num_rack_types):
                    st.sidebar.markdown(f"#### Rack Type {i+1}")
                    r_name = st.sidebar.text_input(f"Rack Name", f"Type {chr(65+i)}", key=f"rn_{i}")
                    r_dim = st.sidebar.text_input(f"Rack Dimensions", "2400x800", key=f"rd_{i}")
                    r_levels = st.sidebar.multiselect(f"Levels", ['A','B','C','D','E','F'], default=['A','B','C','D'], key=f"rl_{i}")
                    caps = {c: st.sidebar.number_input(f"{c} per shelf", 1, 50, 4, key=f"cap_{i}_{c}") for c in unique_c}
                    rack_templates[r_name] = {'levels': r_levels, 'capacities': caps, 'dims': r_dim}

            if st.button("🚀 Generate PDF Labels", type="primary"):
                status = st.empty()
                if generation_method == "By Cell Dimension":
                    df_a = generate_station_wise_assignment(df, base_rack_id, levels, num_cells, bin_rules, status)
                else:
                    df_a = generate_by_rack_type(df, base_rack_id, rack_templates, {}, status)
                
                df_final = assign_sequential_location_ids(df_a)
                if not df_final.empty:
                    st.subheader("📊 Rack Allocation Data")
                    ex_buf = io.BytesIO(); df_final.to_excel(ex_buf, index=False, engine='openpyxl'); ex_buf.seek(0)
                    st.download_button(label="📥 Download Excel Allocation", data=ex_buf.getvalue(), file_name="Rack_Allocation.xlsx")
                    
                    prog = st.progress(0)
                    if output_type == "Rack Labels":
                        pdf, sum_df = generate_rack_labels(df_final, prog)
                        st.download_button("📥 Download Rack Labels PDF", pdf, "Rack_Labels.pdf")
                    elif output_type == "Bin Labels":
                        mtm_models = [m.strip() for m in [model1, model2, model3] if m.strip()]
                        pdf, _ = generate_bin_labels(df_final, mtm_models, prog, status)
                        st.download_button("📥 Download Bin Labels PDF", pdf, "Bin_Labels.pdf")
                    elif output_type == "Rack List":
                        pdf, count = generate_rack_list_pdf(df_final, base_rack_id, None, 4.0, 1.5, "Image.png", prog, status)
                        st.download_button("📥 Download Rack List PDF", pdf, "Rack_List.pdf")
                    prog.empty(); status.empty()
        else:
            st.error("❌ Missing columns.")

if __name__ == "__main__":
    main()
