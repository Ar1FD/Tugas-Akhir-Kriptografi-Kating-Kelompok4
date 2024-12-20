# Tugas-Akhir-Kriptografi-Kating-Kelompok4

## Anggota Kelompok
  - Burhan Ahmad			            4611421026
  - Robert Panca R. Simanjuntak	  4611421040	
  - Bagus Al Qohar 			          4611421057
  - Ari Farhansyah Diraja		      4611421112

## Aplikasi Analisis Sbox

## Deskripsi
Aplikasi ini adalah GUI sederhana untuk melakukan analisis terhadap Sbox dengan beberapa metode.

## Fitur
 - Nonlinearity (NL)
 - Strict Avalanche Criterion (SAC)
 - Linear Approximation Probability (LAP)
 - Differential Approximation Probability (DAP)
 - BIC-SAC
 - BIC-NL

## Persyaratan Sistem
- **Python 3.11** harus diinstal pada sistem.
- **Perpustakaan Python** yang dibutuhkan:
  - `flask`
  - `numpy`
  - `pandas`
  - `io`

## Cara Menjalankan Program

### 1. Clone Repository
Clone repository dari GitHub ke komputer lokal Anda.
```bash
git clone https://github.com/Ar1FD/Tugas-Akhir-Kriptografi-Kating-Kelompok4.git
```

## 2. Instalasi Perpustakaan yang Dibutuhkan
Download perpustakaan yang diperlukan di file `requirement.txt`. Jika belum, install dengan menjalankan perintah berikut di terminal:
```bash
pip install -r requirements.txt
```

## 3. Menjalankan Program
Untuk menjalankan aplikasi, navigasikan ke direktori tempat program disimpan dan jalankan perintah berikut:
```bash
python .\flaskapp.py
```

## 4. Cara Menggunakan Aplikasi
- Upload file sbox yang diinginkan
- Pilih salah satu atau beberapa operasi yang tersedia
- Klik Jalankan Analisis
- Lalu dapat dipilih untuk melakukan download terhadap hasil atau tidak

## Catatan
- Untuk melakukan analisis ulang diperlukan mengupload sbox kembali dan memilih operasi yang diinginkan

