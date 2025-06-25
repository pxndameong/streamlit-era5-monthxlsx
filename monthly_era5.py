import streamlit as st
import pandas as pd
import xarray as xr
import os
import zipfile
from io import BytesIO

# --- Functions from your original script, adapted for Streamlit ---

def check_missing_values(data, label, year=None, month=None):
    """
    Check for NaN/missing values in the data and return detailed information.
    """
    missing_info = data.isna().sum()
    total_missing = missing_info.sum()
    
    messages = []
    if total_missing > 0:
        messages.append(f"⚠️ **Peringatan:** Nilai yang hilang terdeteksi di {label}.")
        if year and month:
            messages.append(f"Tahun: {year}, Bulan: {month}")
        messages.append("Detail nilai yang hilang:")
        for col, count in missing_info[missing_info > 0].items():
            messages.append(f"- **{col}**: {count} hilang")
    else:
        messages.append(f"✅ Tidak ada nilai yang hilang di {label}.")
    return "\n".join(messages)

def get_detailed_nan_info(df, initial_label=""):
    """
    Get detailed NaN info by month/year.
    """
    nan_details = []
    if df.empty:
        return nan_details

    # Ensure 'year' and 'month' columns exist
    if 'valid_time' in df.columns:
        df_temp = df.copy()
        df_temp['year'] = df_temp['valid_time'].dt.year
        df_temp['month'] = df_temp['valid_time'].dt.month
    else:
        # If valid_time not present (e.g., after groupby in loop), skip detailed report
        return nan_details

    grouped_by_month = df_temp.groupby(['year', 'month'])
    
    for (year, month), group_df in grouped_by_month:
        missing_info = group_df.isna().sum()
        missing_cols = missing_info[missing_info > 0]
        
        if not missing_cols.empty:
            detail_str = f"**{initial_label} {year}-{month:02d}**: "
            col_details = []
            for col, count in missing_cols.items():
                col_details.append(f"{col}: {count} NaN")
            detail_str += ", ".join(col_details)
            nan_details.append(detail_str)
            
    return nan_details

@st.cache_data
def _get_sample_df_columns(file_content_buffer, is_pressure_level):
    """
    Helper function to extract time range and column names from a single NetCDF file.
    It mimics the column transformation for pressure level data to ensure
    detected columns match the processed output (e.g., 't_200').
    Returns (min_time, max_time, list_of_columns).
    """
    try:
        file_content_buffer.seek(0) # Ensure buffer is at the start
        with xr.open_dataset(file_content_buffer) as ds:
            if 'valid_time' not in ds.coords and 'valid_time' not in ds.data_vars:
                st.warning(f"Variabel 'valid_time' tidak ditemukan dalam sampel file. Melewatkan file ini untuk metadata.")
                return None, None, []
            
            # Ensure valid_time is datetime
            ds['valid_time'] = pd.to_datetime(ds['valid_time'].values)

            # Perbaikan: Menggunakan .item() untuk mendapatkan nilai skalar dari DataArray, lalu konversi ke datetime
            min_time = pd.to_datetime(ds['valid_time'].min().item())
            max_time = pd.to_datetime(ds['valid_time'].max().item())

            # Cek koordinat tekanan yang mungkin ada di file NetCDF
            pressure_level_coords = ["level", "pressure_level", "isobaricInhPa"]
            actual_pressure_col = None
            for col_name in pressure_level_coords:
                if col_name in ds.coords or col_name in ds.data_vars: # Cek di coords atau data_vars
                    actual_pressure_col = col_name
                    break
            
            all_cols_for_sample = set()
            
            if is_pressure_level and actual_pressure_col:
                # Dapatkan semua tingkat tekanan yang unik
                pressure_levels = ds[actual_pressure_col].values.tolist()
                
                # Iterasi melalui semua variabel data di dataset
                for var_name in ds.data_vars:
                    # Kecualikan variabel yang bukan data yang ingin diproses (misalnya koordinat, expver, number)
                    if var_name not in ['latitude', 'longitude', 'valid_time', 'expver', 'number'] and var_name not in pressure_level_coords:
                        # Untuk setiap variabel data, buat nama kolom untuk setiap level tekanan
                        for level in pressure_levels:
                            all_cols_for_sample.add(f"{var_name}_{int(level)}")
            else: # Single level data atau pressure level tanpa kolom tekanan yang jelas
                for var_name in ds.data_vars:
                    if var_name not in ['latitude', 'longitude', 'valid_time', "expver", "number"] and var_name not in pressure_level_coords:
                        all_cols_for_sample.add(var_name)
            
            # Tangani konversi viwvd ke vimfc untuk deteksi kolom
            if "viwvd" in all_cols_for_sample:
                all_cols_for_sample.remove("viwvd")
                all_cols_for_sample.add("vimfc")
            
            return min_time, max_time, sorted(list(all_cols_for_sample)) # Return sorted unique columns
    except Exception as e:
        st.warning(f"Error membaca sampel file untuk metadata: {e}")
        # Jika ada error, reset status cache data untuk fungsi ini agar bisa di-run lagi jika ada perbaikan
        _get_sample_df_columns.clear() 
        return None, None, []

@st.cache_data
def process_netcdf_data(uploaded_file_contents, is_pressure_level, years, selected_columns):
    """
    Processes uploaded NetCDF file contents.
    `uploaded_file_contents` expected to be a list of BytesIO objects.
    `selected_columns` are the columns to retain in the final output.
    """
    all_data = []
    pressure_level_coords = ["level", "pressure_level", "isobaricInhPa"] 

    for file_content_buffer in uploaded_file_contents:
        try:
            file_content_buffer.seek(0)
            with xr.open_dataset(file_content_buffer) as ds:
                if 'valid_time' not in ds.coords and 'valid_time' not in ds.data_vars:
                    st.warning(f"Variabel 'valid_time' tidak ditemukan di salah satu file. Melewatkan file ini.")
                    continue
                
                ds['valid_time'] = pd.to_datetime(ds['valid_time'].values)
                # Filter by selected years early to reduce memory usage for large files
                ds = ds.sel(valid_time=ds['valid_time'].dt.year.isin(years))
                ds = ds.drop_vars(["expver", "number"], errors="ignore")

                df = ds.to_dataframe().reset_index()

                actual_pressure_col = None
                if is_pressure_level:
                    for col_name in pressure_level_coords:
                        if col_name in df.columns:
                            actual_pressure_col = col_name
                            break

                if actual_pressure_col: 
                    pressure_levels = df[actual_pressure_col].unique()
                    level_dfs = []
                    for level in pressure_levels:
                        level_df = df[df[actual_pressure_col] == level].copy()
                        for var in ds.data_vars:
                            if var in level_df.columns and var not in ['latitude', 'longitude', 'valid_time', actual_pressure_col]: 
                                level_df.rename(columns={var: f"{var}_{int(level)}"}, inplace=True)
                        level_dfs.append(level_df.drop(columns=[actual_pressure_col]))
                    
                    if level_dfs:
                        all_data.append(pd.concat(level_dfs, ignore_index=True))
                else:
                    all_data.append(df)
        except Exception as e:
            st.error(f"Error memproses file NetCDF: {e}")
            continue 

    if not all_data:
        return pd.DataFrame()

    combined_initial_df = pd.concat(all_data, ignore_index=True)
    
    # Tangani konversi viwvd ke vimfc
    if "viwvd" in combined_initial_df.columns:
        combined_initial_df["vimfc"] = combined_initial_df["viwvd"] * -1
        combined_initial_df.drop(columns=["viwvd"], inplace=True)

    # Siapkan kolom akhir untuk dipertahankan
    essential_coords = ['latitude', 'longitude', 'valid_time']
    # Jika 'viwvd' awalnya dipilih, pastikan 'vimfc' ada di final_cols_candidate
    final_cols_candidate = list(selected_columns)
    if "viwvd" in selected_columns and "vimfc" not in final_cols_candidate:
        final_cols_candidate.append("vimfc")
    
    # Pastikan koordinat penting selalu disertakan untuk penggabungan/pengelompokan
    for col in essential_coords:
        if col not in final_cols_candidate:
            final_cols_candidate.append(col)

    # Filter DataFrame hanya untuk menyertakan kolom yang ada dan dipilih/penting
    cols_to_keep_in_df = [col for col in final_cols_candidate if col in combined_initial_df.columns]
    
    return combined_initial_df[cols_to_keep_in_df]

# --- Streamlit UI ---

st.set_page_config(page_title="ERA5 Data Processor", layout="wide")

st.title("🌍 ERA5 Data Processor (NetCDF ke Excel)")

st.markdown("""
Aplikasi ini memungkinkan Anda mengunggah file NetCDF (data ERA5),
memprosesnya, dan mengunduh hasilnya dalam format Excel (.xlsx) per bulan.
Aplikasi ini akan **mengisi (interpolasi) nilai yang hilang (NaN)**.
""")

st.markdown("---")

## 📥 Unggah File NetCDF

col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Tingkat Tunggal (Single Level)")
    uploaded_single_level_files = st.file_uploader(
        "Unggah file NetCDF (single level) Anda di sini (.nc)",
        type=["nc"],
        accept_multiple_files=True,
        key="single_level_uploader"
    )

with col2:
    st.subheader("Data Tingkat Tekanan (Pressure Level)")
    uploaded_pressure_level_files = st.file_uploader(
        "Unggah file NetCDF (pressure level) Anda di sini (.nc)",
        type=["nc"],
        accept_multiple_files=True,
        key="pressure_level_uploader"
    )

st.markdown("---")

# Inisialisasi variabel session state untuk alur UI dan persistensi data
if 'checked_data_available' not in st.session_state:
    st.session_state.checked_data_available = False
if 'detected_min_year' not in st.session_state:
    st.session_state.detected_min_year = None
if 'detected_max_year' not in st.session_state:
    st.session_state.detected_max_year = None
if 'detected_min_month' not in st.session_state:
    st.session_state.detected_min_month = None
if 'detected_max_month' not in st.session_state:
    st.session_state.detected_max_month = None
if 'detected_columns' not in st.session_state:
    st.session_state.detected_columns = []
if 'selected_columns' not in st.session_state:
    st.session_state.selected_columns = []
if 'start_year_input' not in st.session_state:
    st.session_state.start_year_input = None
if 'end_year_input' not in st.session_state:
    st.session_state.end_year_input = None

all_uploaded_files = uploaded_single_level_files + uploaded_pressure_level_files

# --- Tombol "Cek Dulu" ---
if all_uploaded_files:
    if st.button("Cek Dulu Data", key="check_data_button"):
        with st.spinner("Menganalisis file untuk menentukan rentang waktu dan kolom..."):
            min_overall_time = None
            max_overall_time = None
            all_detected_cols_set = set()

            # Proses file tingkat tunggal untuk metadata
            for file_buffer in uploaded_single_level_files:
                # Gunakan BytesIO(file_buffer.getvalue()) untuk membuat buffer baru agar cache berfungsi dengan benar
                min_time, max_time, cols = _get_sample_df_columns(BytesIO(file_buffer.getvalue()), is_pressure_level=False)
                if min_time and max_time:
                    if min_overall_time is None or min_time < min_overall_time:
                        min_overall_time = min_time
                    if max_overall_time is None or max_time > max_overall_time:
                        max_overall_time = max_time
                    all_detected_cols_set.update(cols)
                
            # Proses file tingkat tekanan untuk metadata
            for file_buffer in uploaded_pressure_level_files:
                min_time, max_time, cols = _get_sample_df_columns(BytesIO(file_buffer.getvalue()), is_pressure_level=True)
                if min_time and max_time:
                    if min_overall_time is None or min_time < min_overall_time:
                        min_overall_time = min_time
                    if max_overall_time is None or max_time > max_overall_time:
                        max_overall_time = max_time
                    all_detected_cols_set.update(cols)

            if min_overall_time and max_overall_time:
                st.session_state.detected_min_year = min_overall_time.year
                st.session_state.detected_max_year = max_overall_time.year
                st.session_state.detected_min_month = min_overall_time.month
                st.session_state.detected_max_month = max_overall_time.month
                st.session_state.detected_columns = sorted(list(all_detected_cols_set))
                st.session_state.selected_columns = st.session_state.detected_columns # Pra-pilih semua kolom yang terdeteksi
                st.session_state.checked_data_available = True
                st.success("Analisis data awal selesai! Silakan sesuaikan pengaturan dan klik 'Diproses Data'.")
                st.rerun() # Jalankan ulang untuk menampilkan elemen UI berikutnya segera
            else:
                st.warning("Tidak dapat menentukan rentang waktu atau kolom dari file yang diunggah. Pastikan file valid dan memiliki variabel 'valid_time'.")
                st.session_state.checked_data_available = False

# --- Elemen UI ditampilkan setelah "Cek Dulu" dilakukan ---
if st.session_state.checked_data_available:
    st.markdown("---")
    st.subheader("Informasi Data Terdeteksi dan Pengaturan Pemrosesan")

    if st.session_state.detected_min_year and st.session_state.detected_max_year:
        st.info(f"**Rentang Waktu Terdeteksi:** "
                f"Dari {st.session_state.detected_min_month:02d}/{st.session_state.detected_min_year} "
                f"hingga {st.session_state.detected_max_month:02d}/{st.session_state.detected_max_year}")
        st.markdown("---")

        st.subheader("Pengaturan Tahun")
        current_year = pd.Timestamp.now().year
        
        # Set nilai awal untuk input angka berdasarkan rentang yang terdeteksi, tetapi izinkan pengguna untuk mengubah
        st.session_state.start_year_input = st.number_input(
            "Tahun Mulai", 
            min_value=1940, 
            max_value=current_year, 
            value=st.session_state.detected_min_year if st.session_state.start_year_input is None else st.session_state.start_year_input, 
            step=1, 
            key="start_year_selector"
        )
        st.session_state.end_year_input = st.number_input(
            "Tahun Akhir", 
            min_value=1940, 
            max_value=current_year, 
            value=st.session_state.detected_max_year if st.session_state.end_year_input is None else st.session_state.end_year_input, 
            step=1, 
            key="end_year_selector"
        )

        if st.session_state.start_year_input > st.session_state.end_year_input:
            st.error("Tahun mulai tidak boleh lebih besar dari tahun akhir.")
            
        st.markdown("---")
        st.subheader("Pilih Kolom untuk Disertakan dalam Output")
        st.session_state.selected_columns = st.multiselect(
            "Pilih kolom data yang ingin Anda sertakan (selain lintang, bujur, dan waktu):",
            options=st.session_state.detected_columns,
            default=st.session_state.selected_columns,
            key="column_selector"
        )

        st.markdown("---")
        
        # --- Tombol "Diproses Data" ---
        if st.button("Diproses Data", key="process_data_button"):
            if not all_uploaded_files:
                st.warning("Mohon unggah setidaknya satu file NetCDF untuk memulai pemrosesan.")
            elif not st.session_state.selected_columns:
                st.warning("Mohon pilih setidaknya satu kolom untuk diproses.")
            elif st.session_state.start_year_input > st.session_state.end_year_input:
                st.error("Tahun mulai tidak valid. Mohon periksa kembali.")
            else:
                years_to_process = list(range(st.session_state.start_year_input, st.session_state.end_year_input + 1))
                st.info(f"Memproses data untuk tahun: **{', '.join(map(str, years_to_process))}** "
                        f"dan kolom yang dipilih: **{', '.join(st.session_state.selected_columns)}**")

                single_level_contents = [BytesIO(f.getvalue()) for f in uploaded_single_level_files]
                pressure_level_contents = [BytesIO(f.getvalue()) for f in uploaded_pressure_level_files]

                df_single_level = pd.DataFrame()
                df_pressure_level = pd.DataFrame()

                with st.spinner("Memproses data Tingkat Tunggal (Single Level)..."):
                    df_single_level = process_netcdf_data(single_level_contents, is_pressure_level=False, years=years_to_process, selected_columns=st.session_state.selected_columns)
                    if not df_single_level.empty:
                        st.subheader("Analisis Missing Values (Single Level Data - Sebelum Imputasi):")
                        st.markdown(check_missing_values(df_single_level, "Single Level Data"))
                        detailed_nan_single = get_detailed_nan_info(df_single_level, "Single Level Data -")
                        if detailed_nan_single:
                            for detail in detailed_nan_single:
                                st.markdown(f"- {detail}")
                        else:
                            st.info("Tidak ada nilai yang hilang di Single Level Data (per bulan).")


                with st.spinner("Memproses data Tingkat Tekanan (Pressure Level)..."):
                    df_pressure_level = process_netcdf_data(pressure_level_contents, is_pressure_level=True, years=years_to_process, selected_columns=st.session_state.selected_columns)
                    if not df_pressure_level.empty:
                        st.subheader("Analisis Missing Values (Pressure Level Data - Sebelum Imputasi):")
                        st.markdown(check_missing_values(df_pressure_level, "Pressure Level Data"))
                        detailed_nan_pressure = get_detailed_nan_info(df_pressure_level, "Pressure Level Data -")
                        if detailed_nan_pressure:
                            for detail in detailed_nan_pressure:
                                st.markdown(f"- {detail}")
                        else:
                            st.info("Tidak ada nilai yang hilang di Pressure Level Data (per bulan).")

                if not df_single_level.empty or not df_pressure_level.empty:
                    st.subheader("Menggabungkan dan Memproses Data...")
                    
                    combined_df = pd.DataFrame()
                    if not df_single_level.empty and not df_pressure_level.empty:
                        # Gunakan 'outer' merge untuk mempertahankan semua titik data, lalu interpolasi NaN
                        combined_df = pd.merge(df_single_level, df_pressure_level, on=["latitude", "longitude", "valid_time"], how="outer")
                    elif not df_single_level.empty:
                        combined_df = df_single_level
                    elif not df_pressure_level.empty:
                        combined_df = df_pressure_level
                    else:
                        st.warning("Tidak ada data yang tersedia untuk digabungkan setelah pemrosesan awal.")
                        st.stop()

                    if combined_df.empty:
                        st.error("DataFrame gabungan kosong setelah merge. Periksa kembali file yang diunggah atau rentang tahun.")
                        st.stop()

                    # --- Imputasi NaN ---
                    st.subheader("Melakukan Imputasi (Mengisi) Nilai yang Hilang...")
                    # Urutkan berdasarkan waktu dan lokasi sebelum interpolasi untuk hasil terbaik
                    combined_df = combined_df.sort_values(by=['valid_time', 'latitude', 'longitude']).reset_index(drop=True)
                    
                    # Identifikasi kolom numerik untuk interpolasi di antara yang dipilih
                    cols_to_interpolate = [col for col in st.session_state.selected_columns if col in combined_df.columns and pd.api.types.is_numeric_dtype(combined_df[col])]
                    
                    if combined_df[cols_to_interpolate].isnull().values.any():
                        combined_df[cols_to_interpolate] = combined_df[cols_to_interpolate].interpolate(method='linear', limit_direction='both', axis=0)
                        st.success("Nilai yang hilang telah diinterpolasi.")
                    else:
                        st.info("Tidak ada nilai yang hilang yang perlu diinterpolasi di kolom yang dipilih.")

                    # --- Analisis Missing Values Setelah Imputasi ---
                    st.subheader("Analisis Missing Values (Setelah Imputasi):")
                    st.markdown(check_missing_values(combined_df, "Data Gabungan (Setelah Imputasi)"))
                    detailed_nan_after_imputation = get_detailed_nan_info(combined_df, "Data Gabungan - Setelah Imputasi")
                    if detailed_nan_after_imputation:
                        for detail in detailed_nan_after_imputation:
                            st.markdown(f"- {detail}")
                    else:
                        st.info("Tidak ada nilai yang hilang di Data Gabungan setelah imputasi (per bulan).")


                    # Ubah nama kolom untuk konsistensi output akhir
                    combined_df.rename(columns={"latitude": "lat", "longitude": "lon"}, inplace=True)

                    # Kelompokkan berdasarkan tahun dan bulan untuk file Excel bulanan
                    grouped = combined_df.groupby([combined_df['valid_time'].dt.year, combined_df['valid_time'].dt.month])

                    output_zip = BytesIO()
                    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                        progress_bar = st.progress(0)
                        total_groups = len(grouped)
                        
                        if total_groups == 0:
                            st.warning("Tidak ada data yang tersedia untuk diekspor setelah pengelompokan berdasarkan tahun/bulan. Ini mungkin terjadi jika rentang tahun yang dipilih tidak memiliki data.")
                            st.stop()

                        for i, ((year, month), data) in enumerate(grouped):
                            st.write(f"Mengekspor data untuk **{year}-{month:02d}**...")
                            
                            data_to_save = data.copy() # Operasikan pada salinan
                            
                            # Kolom yang akan disimpan untuk Excel akhir, sebelum agregasi terakhir berdasarkan lat/lon
                            # Ini adalah lat, lon, dan semua kolom data yang dipilih pengguna (yang sudah difilter di process_netcdf_data)
                            cols_for_final_excel = ['lat', 'lon'] + st.session_state.selected_columns
                            
                            # Filter data_to_save untuk memastikan hanya kolom yang ada yang disimpan untuk agregasi
                            actual_cols_for_final_excel = [col for col in cols_for_final_excel if col in data_to_save.columns]
                            data_to_save = data_to_save[actual_cols_for_final_excel]

                            # Identifikasi kolom numerik untuk agregasi (kecualikan lat, lon)
                            numeric_cols_to_agg = [col for col in data_to_save.columns if col not in ['lat', 'lon'] and pd.api.types.is_numeric_dtype(data_to_save[col])]
                            
                            if numeric_cols_to_agg:
                                # Lakukan agregasi rata-rata berdasarkan lat/lon untuk kolom data numerik
                                data_to_save = data_to_save.groupby(['lat', 'lon'])[numeric_cols_to_agg].mean().reset_index()
                            else:
                                st.warning(f"Tidak ada kolom numerik yang dipilih atau ditemukan untuk dirata-ratakan pada {year}-{month:02d}. Melewatkan ekspor untuk bulan ini.")
                                continue
                            
                            # Periksa nilai yang hilang SETELAH agregasi (seharusnya minimal jika interpolasi efektif)
                            st.markdown(check_missing_values(data_to_save, f"Data Terproses untuk {year}-{month:02d}", year, month))

                            output_file_name = f"era5jawa_{year}_{month:02d}.xlsx"
                            excel_buffer = BytesIO()
                            data_to_save.to_excel(excel_buffer, index=False)
                            excel_buffer.seek(0) 

                            zf.writestr(output_file_name, excel_buffer.getvalue())
                            progress_bar.progress((i + 1) / total_groups)
                        
                    output_zip.seek(0)
                    st.success("Pemrosesan selesai! File Excel siap diunduh.")
                    
                    st.download_button(
                        label="📥 Unduh Semua File Excel (ZIP)",
                        data=output_zip.getvalue(),
                        file_name=f"era5_data_output_{st.session_state.start_year_input}-{st.session_state.end_year_input}.zip",
                        mime="application/zip"
                    )
                else:
                    st.warning("Tidak ada data yang berhasil diproses. Pastikan file yang diunggah valid dan sesuai dengan tahun yang dipilih.")

st.markdown("---")
st.markdown("Dibuat dengan ❤️ oleh tsaqib v9")