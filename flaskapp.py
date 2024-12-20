from flask import Flask, render_template, request, redirect, send_file, session
import pandas as pd
import numpy as np
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'apa_aja'

# === Fungsi Perhitungan S-Box ===

def validate_sbox(sbox):
    if len(sbox) != 256:
        return False, "S-Box harus memiliki nilai tepat 256."
    if len(set(sbox)) != 256:
        return False, "S-Box tidak valid: memiliki nilai/angka duplikat."
    if any(x < 0 or x > 255 for x in sbox):
        return False, "S-Box memiliki nilai di luar rentang 0-255."
    return True, ""

def calculate_nonlinearity(sbox):
    def walsh_hadamard_transform(f):
        n = 256
        wht = np.zeros(n, dtype=int)
        for a in range(n):
            sum_val = 0
            for x in range(n):
                correlation = (-1) ** (bin(a & x).count('1') % 2)
                sum_val += correlation * ((-1) ** (bin(a & f[x]).count('1') % 2))
            wht[a] = sum_val
        return wht

    wht = walsh_hadamard_transform(sbox)
    nonlinearity = (256 // 2) - max(abs(wht[1:])) // 2
    return nonlinearity

def calculate_sac(sbox):
    sac_results = []
    for bit_pos in range(8):
        change_count = sum(
            bin(sbox[x] ^ sbox[x ^ (1 << bit_pos)]).count('1')
            for x in range(256)
        )
        sac_results.append(change_count / (256 * 8))
    return np.mean(sac_results)

def calculate_lap(sbox):
    max_bias = 0
    for a in range(1, 256):
        for b in range(1, 256):
            correlation_count = sum(
                (bin(a & x).count('1') % 2) == (bin(b & sbox[x]).count('1') % 2)
                for x in range(256)
            )
            bias = abs(correlation_count / 256.0 - 0.5)
            max_bias = max(max_bias, bias)
    return max_bias

def calculate_dap(sbox):
    max_prob = 0
    for delta_in in range(1, 256):
        diff_count = {}
        for x in range(256):
            delta_out = sbox[x ^ delta_in] ^ sbox[x]
            diff_count[delta_out] = diff_count.get(delta_out, 0) + 1
        current_max_prob = max(count / 256.0 for count in diff_count.values())
        max_prob = max(max_prob, current_max_prob)
    return max_prob

# Fungsi BIC-SAC
def calculate_bic_sac(sbox):
    n = len(sbox)
    bit_length = 8
    total_pairs = 0
    total_independence = 0
    for i in range(bit_length):
        for j in range(i + 1, bit_length):
            independence_sum = 0
            for x in range(n):
                for bit_to_flip in range(bit_length):
                    flipped_x = x ^ (1 << bit_to_flip)
                    y1 = sbox[x]
                    y2 = sbox[flipped_x]
                    b1_i = (y1 >> i) & 1
                    b1_j = (y1 >> j) & 1
                    b2_i = (y2 >> i) & 1
                    b2_j = (y2 >> j) & 1
                    independence_sum += ((b1_i ^ b2_i) ^ (b1_j ^ b2_j))
            total_independence += independence_sum / (n * bit_length)
            total_pairs += 1
    return round(total_independence / total_pairs, 5)

# Fungsi BIC-NL
def calculate_bic_nl(sbox):
    def binary_representation(num, bits=8):
        return np.array([int(b) for b in format(num, f'0{bits}b')])

    def optimized_walsh_hadamard(sbox, n=8, m=8):
        inputs = np.array([binary_representation(x, n) for x in range(2**n)])
        outputs = np.array([binary_representation(sbox[x], m) for x in range(2**n)])
        max_walsh = 0
        for u in range(1, 2**n):
            u_bin = binary_representation(u, n)
            for v in range(1, 2**m):
                v_bin = binary_representation(v, m)
                dot_u_x = (inputs @ u_bin) % 2
                dot_v_Sx = (outputs @ v_bin) % 2
                dot_result = (dot_u_x ^ dot_v_Sx)
                walsh_sum = np.sum(1 - 2 * dot_result)
                max_walsh = max(max_walsh, abs(walsh_sum))
        return 2**(n-1) - max_walsh / 2

    n = 8
    total_nl = 0
    total_pairs = 0
    for bit1 in range(n):
        for bit2 in range(n):
            if bit1 != bit2:
                combined_sbox = [(sbox[x] >> bit1 & 1) ^ (sbox[x] >> bit2 & 1) for x in range(256)]
                nl = optimized_walsh_hadamard(combined_sbox, n=8, m=1)
                total_nl += nl
                total_pairs += 1
    return round(total_nl / total_pairs, 5)

def calculate_sac_matrix(sbox):
    sac_matrix = np.zeros((8, 8)) 
    for input_bit in range(8):  
        for output_bit in range(8):  
            change_count = 0
            for x in range(256):  
                flipped_x = x ^ (1 << input_bit)
                original_output = (sbox[x] >> output_bit) & 1
                flipped_output = (sbox[flipped_x] >> output_bit) & 1
                if original_output != flipped_output:
                    change_count += 1
            sac_matrix[input_bit][output_bit] = change_count / 256
    return sac_matrix

# === Flask Routes ===

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    try:
        df = pd.read_excel(file, header=None)
        sbox = df.values.flatten().tolist()
    except Exception as e:
        return render_template('index.html', error_message=f"Error membaca file: {e}")

    # Validate S-Box
    is_valid, error_message = validate_sbox(sbox)
    if not is_valid:
        return render_template('index.html', error_message=error_message)

    # Store the S-Box in session
    session['sbox_data'] = [int(x) for x in sbox]

    # Get selected operations from the form
    selected_operations = request.form.getlist('operations')  # Get the selected checkboxes

    # Perform selected operations
    results = {}
    if 'Nonlinearity' in selected_operations:
        results["Nonlinearity (NL)"] = calculate_nonlinearity(sbox)
    if 'SAC' in selected_operations:
        results["Strict Avalanche Criterion (SAC)"] = calculate_sac(sbox)
    if 'LAP' in selected_operations:
        results["Linear Approximation Probability (LAP)"] = calculate_lap(sbox)
    if 'DAP' in selected_operations:
        results["Differential Approximation Probability (DAP)"] = calculate_dap(sbox)
    if 'BIC-SAC' in selected_operations:
        results["BIC-SAC"] = calculate_bic_sac(sbox)
    if 'BIC-NL' in selected_operations:
        results["BIC-NL"] = calculate_bic_nl(sbox) 

    sac_matrix_df = None
    if 'SAC' in selected_operations:
        sac_matrix = calculate_sac_matrix(sbox)
        sac_matrix_df = pd.DataFrame(
            sac_matrix,
            columns=[f"Output Bit {i}" for i in range(8)],
            index=[f"Input Bit {i}" for i in range(8)]
        )

    # Prepare results for display
    results_df = pd.DataFrame.from_dict(results, orient="index", columns=["Nilai"])
    sac_matrix_html = sac_matrix_df.to_html(classes="table table-striped") if sac_matrix_df is not None else None

    # Render the results with selected operations to keep checkboxes checked
    return render_template('index.html', results_df=results_df.to_html(classes="table table-bordered"),
                           sac_matrix_html=sac_matrix_html, download_results=True, sbox_data=sbox,
                           selected_operations=selected_operations)  # Pass selected_operations to template


@app.route('/download', methods=['POST'])
def download():
    try:
        # Ambil data S-Box dari sesi
        sbox_data = session.get('sbox_data')
        if not sbox_data:
            return redirect('/')

        # Ambil operasi yang dipilih dari form
        selected_operations = request.form.getlist('operations[]')

        # Pastikan ada operasi yang dipilih
        if not selected_operations:
            return render_template('index.html', error_message="Tidak ada analisis yang dipilih.")

        # Daftar operasi
        operations = {
            "Nonlinearity": calculate_nonlinearity,
            "SAC": calculate_sac,  # Gunakan hasil hitung, bukan matrix
            "LAP": calculate_lap,
            "DAP": calculate_dap,
            "BIC-SAC": calculate_bic_sac,
            "BIC-NL": calculate_bic_nl
        }

        # Jalankan setiap operasi yang dipilih dan kumpulkan hasil
        results = {}

        for operation in selected_operations:
            if operation in operations:
                results[operation] = operations[operation](sbox_data)

        # Siapkan file untuk diunduh
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            # Tuliskan hasil utama (Summary)
            if results:
                results_df = pd.DataFrame.from_dict(results, orient="index", columns=["Nilai"])
                results_df.to_excel(writer, sheet_name="Summary", index_label="Analisis")

        # Kirimkan file ke pengguna
        output.seek(0)
        return send_file(
            output,
            as_attachment=True,
            download_name="sbox_analysis_results.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        return render_template('index.html', error_message=f"Error saat mendownload file: {e}")

if __name__ == '__main__':
    app.run(debug=True)

