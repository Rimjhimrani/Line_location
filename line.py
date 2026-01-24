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
    station_no_key = next((k for k in cols if 'STATION' in k), None)
    container_type_key = next((k for k in cols if 'CONTAINER' in k), None)
    return {
        'Part No': cols.get(part_no_key),
        'Description': cols.get(desc_key),
        'Bus Model': cols.get(bus_model_key),
        'Station No': cols.get(station_no_key),
        'Container': cols.get(container_type_key)
    }

def get_unique_containers(df, container_col):
    if not container_col or container_col not in df.columns: return []
    return sorted(df[container_col].dropna().astype(str).unique())

def parse_dimensions(dim_str):
    if not isinstance(dim_str, str) or not dim_str: return 0, 0
    nums = [int(n) for n in re.findall(r'\d+', dim_str)]
    return (nums[0], nums[1]) if len(nums) >= 2 else (0, 0)

def generate_station_wise_assignment(df, base_rack_id, levels, cells_per_level, bin_info_map, status_text=None):
    required_cols = find_required_columns(df)
    part_no_col = required_cols['Part No']
    desc_col = required_cols['Description']
    model_col = required_cols['Bus Model']
    station_col = required_cols['Station No']
    container_col = required_cols['Container']
    
    df_processed = df.copy()
    rename_dict = {
        part_no_col: 'Part No', desc_col: 'Description',
        model_col: 'Bus Model', station_col: 'Station No', container_col: 'Container'
    }
    df_processed.rename(columns={k: v for k, v in rename_dict.items() if k}, inplace=True)
    
    df_processed['bin_info'] = df_processed['Container'].map(bin_info_map)
    df_processed['bin_area'] = df_processed['bin_info'].apply(lambda x: x['dims'][0] * x['dims'][1] if x else 0)
    df_processed['bins_per_cell'] = df_processed['bin_info'].apply(lambda x: x['capacity'] if x else 1)
    
    final_assigned_data = []

    for station_no, station_group in df_processed.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Processing Station: {station_no}...")
        
        station_cells_needed = 0
        container_groups = sorted(station_group.groupby('Container'), key=lambda x: x[1]['bin_area'].iloc[0], reverse=True)
        
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
                loc = station_available_cells[current_cell_ptr]
                for p in chunk:
                    p.update(loc)
                    final_assigned_data.append(p)
                current_cell_ptr += 1
        
        for i in range(current_cell_ptr, len(station_available_cells)):
            empty_label = {
                'Part No': 'EMPTY', 'Description': '', 'Bus Model': station_group['Bus Model'].iloc[0], 
                'Station No': station_no, 'Container': ''
            }
            empty_label.update(station_available_cells[i])
            final_assigned_data.append(empty_label)

    return pd.DataFrame(final_assigned_data)

def generate_by_rack_type(df, base_rack_id, rack_configs, container_dims, status_text=None):
    """Generate assignments using predefined rack types with automated rack allocation"""
    required_cols = find_required_columns(df)
    part_no_col = required_cols['Part No']
    desc_col = required_cols['Description']
    model_col = required_cols['Bus Model']
    station_col = required_cols['Station No']
    container_col = required_cols['Container']
    
    df_processed = df.copy()
    rename_dict = {
        part_no_col: 'Part No', desc_col: 'Description',
        model_col: 'Bus Model', station_col: 'Station No', container_col: 'Container'
    }
    df_processed.rename(columns={k: v for k, v in rename_dict.items() if k}, inplace=True)
    
    final_parts_list = []
    
    for station_no, station_group in df_processed.groupby('Station No', sort=False):
        if status_text: status_text.text(f"Processing station: {station_no}...")
        
        rack_idx, level_idx = 0, 0
        sorted_racks = sorted(rack_configs.items())
        
        for container_type, parts_group in station_group.groupby('Container', sort=True):
            items_to_place = parts_group.to_dict('records')
            
            while items_to_place:
                slot_found = False
                search_rack_idx, search_level_idx = rack_idx, level_idx
                
                while search_rack_idx < len(sorted_racks):
                    rack_name, config = sorted_racks[search_rack_idx]
                    levels, capacity = config.get('levels', []), config.get('rack_bin_counts', {}).get(container_type, 0)
                    
                    if capacity > 0 and search_level_idx < len(levels):
                        slot_found = True
                        rack_idx, level_idx = search_rack_idx, search_level_idx
                        break
                    
                    search_level_idx = 0
                    search_rack_idx += 1
                
                if not slot_found:
                    if status_text: status_text.warning(f"⚠️ Ran out of rack space at Station {station_no} for '{container_type}'.")
                    break
                
                rack_name, config = sorted_racks[rack_idx]
                levels = config.get('levels', [])
                level_capacity = config.get('rack_bin_counts', {}).get(container_type, 0)
                parts_for_level = items_to_place[:level_capacity]
                items_to_place = items_to_place[level_capacity:]
                
                num_empty_slots = level_capacity - len(parts_for_level)
                level_items = parts_for_level + ([{'Part No': 'EMPTY'}] * num_empty_slots)
                
                item_template = {col: '' for col in df_processed.columns}
                
                for cell_idx, item in enumerate(level_items, 1):
                    rack_num_val = ''.join(filter(str.isdigit, rack_name))
                    rack_num_1st = rack_num_val[0] if len(rack_num_val) > 1 else '0'
                    rack_num_2nd = rack_num_val[1] if len(rack_num_val) > 1 else rack_num_val[0]
                    
                    location_info = {
                        'Rack': base_rack_id, 'Rack No 1st': rack_num_1st, 'Rack No 2nd': rack_num_2nd,
                        'Level': levels[level_idx], 'Cell': str(cell_idx), 'Station No': station_no,
                        'Rack Key': f"{rack_num_1st}{rack_num_2nd}"
                    }
                    
                    if item['Part No'] == 'EMPTY':
                        full_item = item_template.copy()
                        full_item.update({'Part No': 'EMPTY', 'Container': container_type})
                    else:
                        full_item = item
                    
                    full_item.update(location_info)
                    final_parts_list.append(full_item)
                
                level_idx += 1
                if level_idx >= len(levels):
                    level_idx = 0
                    rack_idx += 1
    
    if not final_parts_list: 
        return pd.DataFrame()
    
    return pd.DataFrame(final_parts_list)
    
def assign_sequential_location_ids(df):
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
    station_name = get_clean_value(['ST. NAME (Short)', 'ST.NAME (Short)', 'ST NAME (Short)', 'ST. NAME', 'Station Name Short', 'Station Name']) 
    
    return [station_name, store_location, zone, location, floor, rack_no, level_in_rack]

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
        if status_text: status_text.text(f"Processing Bin Label {i+1}/{total_labels}")
        
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

# --- PDF Generation: Rack List ---
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

        designed_by_text = Paragraph("Designed by:", ParagraphStyle('DesignedBy', fontName='Helvetica', fontSize=10, alignment=TA_CENTER))
        right_inner_table = Table([[designed_by_text, fixed_logo_img]], colWidths=[3*cm, 4.5*cm])
        right_inner_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (0,0), (-1,-1), 'RIGHT'),
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

# --- Main Application UI ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("<p style='font-style:italic;'>Designed and Developed by Rimjhim Rani | Agilomatrix</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.sidebar.title("📄 Configuration")
    
    # Output Type Selection
    output_type = st.sidebar.selectbox("Choose Output Type:", ["Rack Labels", "Bin Labels", "Rack List"])
    
    # Top Logo Upload for Rack List
    top_logo_file = None
    top_logo_w, top_logo_h = 4.0, 1.5
    if output_type == "Rack List":
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Rack List Configuration**")
        top_logo_file = st.sidebar.file_uploader("Upload Top Logo (Optional)", type=['png', 'jpg', 'jpeg'])
        if top_logo_file:
            c1, c2 = st.sidebar.columns(2)
            top_logo_w = c1.slider("Logo Width (cm)", 1.0, 8.0, 4.0)
            top_logo_h = c2.slider("Logo Height (cm)", 0.5, 4.0, 1.5)
    
    # MTM Models for Bin Labels
    model1, model2, model3 = "", "", ""
    if output_type == "Bin Labels":
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Vehicle Models for MTM Table**")
        model1 = st.sidebar.text_input("Vehicle Model 1", value="7M")
        model2 = st.sidebar.text_input("Vehicle Model 2", value="9M")
        model3 = st.sidebar.text_input("Vehicle Model 3", value="12M")
    
    st.sidebar.markdown("---")
    base_rack_id = st.sidebar.text_input("Infrastructure ID", "R")

    # ADD THIS SECTION HERE:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Generation Method")
    generation_method = st.sidebar.radio(
        "Choose Method:",
        ["By Cell Dimension", "By Rack Type"],
        help="Cell Dimension: Traditional uniform rack setup | Rack Type: Predefined rack configurations"
    )
    
    uploaded_file = st.file_uploader("Upload Your Data Here (Excel/CSV)", type=['xlsx', 'xls', 'csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"✅ Loaded {len(df)} parts.")
        
        required_cols = find_required_columns(df)
        
        if required_cols['Container'] and required_cols['Station No']:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Global Rack Settings")
            cell_dim = st.sidebar.text_input("Cell Dimensions (L x W)", "800x400")
            levels = st.sidebar.multiselect("Active Levels", options=['A','B','C','D','E','F','G','H'], default=['A','B','C','D'])
            num_cells_per_level = st.sidebar.number_input("Cells per Level", min_value=1, value=10)
            
            unique_containers = get_unique_containers(df, required_cols['Container'])
            bin_info_map = {}
            st.sidebar.markdown("---")
            st.sidebar.subheader("Container Rules")
            for container in unique_containers:
                st.sidebar.markdown(f"**{container}**")
                dim = st.sidebar.text_input(f"Dimensions", key=f"d_{container}", placeholder="600x400")
                cap = st.sidebar.number_input("Parts per Physical Cell (Capacity)", min_value=1, value=1, key=f"c_{container}")
                bin_info_map[container] = {'dims': parse_dimensions(dim), 'capacity': cap}

            if st.button("🚀 Generate PDF Labels", type="primary"):
                status_text = st.empty()
                
                df_assigned = generate_station_wise_assignment(df, base_rack_id, levels, num_cells_per_level, bin_info_map, status_text)
                df_final = assign_sequential_location_ids(df_assigned)
                
                if not df_final.empty:
                    prog = st.progress(0)
                    
                    if output_type == "Rack Labels":
                        pdf_buf, summary_df = generate_rack_labels(df_final, prog)
                        file_name = "Rack_Labels.pdf"
                        st.download_button(label="📥 Download Rack Labels PDF", data=pdf_buf.getvalue(), file_name=file_name, mime="application/pdf")
                        st.subheader("📊 Generation Summary")
                        st.table(summary_df.sort_values(by=['Station No', 'Rack']))
                    
                    elif output_type == "Bin Labels":
                        mtm_models = [model.strip() for model in [model1, model2, model3] if model.strip()]
                        pdf_buf, label_summary = generate_bin_labels(df_final, mtm_models, prog, status_text)
                        if pdf_buf:
                            file_name = "Bin_Labels.pdf"
                            st.download_button(label="📥 Download Bin Labels PDF", data=pdf_buf.getvalue(), file_name=file_name, mime="application/pdf")
                            st.subheader("📊 Generation Summary")
                            summary_df = pd.DataFrame(list(label_summary.items()), columns=['Location', 'Number of Labels']).sort_values(by='Location').reset_index(drop=True)
                            st.table(summary_df)
                    
                    elif output_type == "Rack List":
                        fixed_logo_path = "Image.png"  # Place your Agilomatrix logo file in the same directory
                        pdf_buf, count = generate_rack_list_pdf(
                            df_final, 
                            base_rack_id, 
                            top_logo_file, 
                            top_logo_w, 
                            top_logo_h, 
                            fixed_logo_path, 
                            prog, 
                            status_text
                        )
                        file_name = "Rack_List.pdf"
                        st.download_button(label="📥 Download Rack List PDF", data=pdf_buf.getvalue(), file_name=file_name, mime="application/pdf")
                        st.success(f"✅ Generated {count} rack lists successfully!")
                    
                    prog.empty()
                status_text.empty()
        else:
            st.error("❌ Missing 'Station' or 'Container' columns.")

if __name__ == "__main__":
    main()
