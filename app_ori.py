from flask import Flask, render_template, request, jsonify
from datetime import datetime
from werkzeug.utils import secure_filename
import mariadb
import sys
import os
import subprocess


app = Flask(__name__, template_folder='templates',
            static_folder='uploads', static_url_path='/uploads')
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['OUTPUT_FOLDER'] = './output_mids/'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # max 20MB
ALLOWED_EXTENSIONS = set(['wav'])

# Connect to MariaDB
# try:
#     conn = mariadb.connect(
#         user="root",
#         password="123456",
#         host="127.0.0.1",
#         port=3306,
#         database="rse_dataset"
#     )
# except mariadb.Error as e:
#     print(f"Error connecting to MariaDB: {e}")
#     sys.exit(1)
# cur = conn.cursor()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def remove_file_extension(filename):
    return os.path.splitext(filename)[0]


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/upload", methods=["POST", "GET"])
def upload():
    file = request.files['uploadFile']

    if file and allowed_file(file.filename):
        input_filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], input_filename))
        input_filename_no_extension = remove_file_extension(input_filename)

        # Run subprocess
        do_denoise = int('denoise_chkbox' in request.form)
        output = subprocess.run(["python", "python_auto_execute.py", "--input",
                                 input_filename_no_extension, "--denoise", str(do_denoise), "--output_path", app.config['OUTPUT_FOLDER']], capture_output=True)
        print(output)

        # Save to database
        current_time = datetime.today()
        statement = "INSERT INTO file_upload (filename, file_path, is_denoised, time_stamp) VALUES (%s, %s, %s, %s)"
        data = (input_filename, os.path.abspath(
            input_filename), False, current_time)
        cur.execute(statement, data)
        conn.commit()

        if do_denoise == 1:
            output_filename_no_post = input_filename_no_extension + \
                "_reduction_visualize_no_post.mid"
            output_filename_final = input_filename_no_extension + \
                "_reduction_visualize_final.mid"
        else:
            output_filename_no_post = input_filename_no_extension + "_visualize_no_post.mid"
            output_filename_final = input_filename_no_extension + "_visualize_final.mid"

        msg = 'Successfully uploaded and processed ' + input_filename
    else:
        msg = 'Invalid file extension type, uplaod only wav files'

    return jsonify({'htmlresponse': render_template('response.html', msg=msg, input_filename=input_filename,
                                                    output1=output_filename_no_post, output2=output_filename_final)})


if __name__ == "__main__":
    app.run(debug=True)


# global system_info_last_row_id
# def write_operation_record(jsonData):
#     time_stamp = jsonData["ts"]
#     operation_mode = jsonData["op"]
#     battery_system_config = jsonData["battery_system_config"]

#     statement = "INSERT INTO Operation_Info (time_stamp, operation_mode, config) VALUES (%s, %s, %s)"
#     data = (time_stamp, operation_mode, battery_system_config)
#     cur.execute(statement, data)
#     conn.commit()
#     system_info_last_row_id = cur.lastrowid
#     return "1"


# def write_battery_record(jsonData):

#     global system_info_last_row_id
#     data_type = jsonData["type"]

#     if data_type == "system":
#         try:
#             op_desc = jsonData["op"]
#             time_stamp = jsonData["ts"]
#             eload_v = jsonData["v"]
#             eload_i = jsonData["i"]
#             eload_p = jsonData["p"]
#             eload_q = jsonData["q"]
#             eload_kwh = jsonData["kwh"]

#             statement = "INSERT INTO System_Info (op_desc, time_stamp, eload_V, eload_I, eload_P, eload_Q, eload_kwh) VALUES (%s, %s, %s, %s, %s, %s, %s)"
#             data = (op_desc, time_stamp, eload_v, eload_i, eload_p, eload_q, eload_kwh)
#             cur.execute(statement, data)
#             conn.commit()

#             system_info_last_row_id = cur.lastrowid
#             print(sys.stdout, "----[NEW RECORD] system_info_id: ", str(system_info_last_row_id))

#         except ValueError as e:
#             print(sys.stderr, "----[ERROR] Insert System_Info: ", e)
#             return -1

#     if data_type == "battery":
#         try:
#             mod_num = jsonData["mod"]
#             module_enby_state = jsonData["enby"]
#             im = jsonData["IM"]
#             vm = jsonData["VM"]
#             ib = jsonData["IB"]
#             vb = jsonData["VB"]
#             t1 = jsonData["T"]
#             gpio1 = jsonData["gpio"][0]
#             gpio2 = jsonData["gpio"][1]
#             gpio3 = jsonData["gpio"][2]
#             gpio4 = jsonData["gpio"][3]
#             gpio5 = jsonData["gpio"][4]
#             # capacity = 0
#             capacity = jsonData["cap"]
#             cell1_v = jsonData["c"][0]
#             cell2_v = jsonData["c"][1]
#             cell3_v = jsonData["c"][2]
#             cell4_v = jsonData["c"][3]

#         except ValueError as e:
#             print(sys.stderr, "----[ERROR] Insert Battery_Info: ", e)
#             return -1

#         statement = "INSERT INTO Battery_Info (module, module_enby_state, vm, vb, im, ib, t1, capacity, gpio1, gpio2, gpio3, gpio4, gpio5, cell1_v, cell2_v, cell3_v, cell4_v, system_info_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
#         data = (mod_num, module_enby_state, vm, vb, im, ib, t1, capacity, gpio1, gpio2, gpio3, gpio4, gpio5, cell1_v, cell2_v, cell3_v, cell4_v, system_info_last_row_id)

#         cur.execute(statement, data)
#         conn.commit()

#     return "1"


# server_address = ("192.168.1.193", 3000)
# sock = socket.socket()
# sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# sock.bind(server_address)
# sock.listen(1)

# incomplete_msg = ""

# while True:
#     print(sys.stdout, "Awaiting connection...")
#     connection, client_address = sock.accept()

#     try:
#         print(sys.stdout, client_address, " connected")

#         while True:
#             # data = connection.recv(8192)
#             data = connection.recv(2048)
#             data = data.decode('utf-8')
#             print(sys.stdout, "Received message: '%s'" % data)

#             if data:
#                 data_array = data.splitlines()
#                 for i in range(len(data_array)):
#                     if len(data_array[i]) > 0:
#                         try:
#                             if incomplete_msg:
#                                 # print("----[IMCOMPLETE]: ", incomplete_msg)
#                                 # print("----[LEFT OVER]: ", data_array[i])
#                                 complete_msg = incomplete_msg + data_array[i]
#                                 print("----[MERGED]: ", complete_msg)
#                                 json_string = json.loads(complete_msg)
#                                 incomplete_msg = ""
#                             else:
#                                 json_string = json.loads(data_array[i])

#                             record_type = json_string["type"]

#                             if record_type == "operation":
#                                 write_operation_record(json_string)
#                             else:
#                                 write_battery_record(json_string)
#                         except ValueError as e:
#                             incomplete_msg = data_array[i]
#                             # print("----[IMCOMPLETE]: ", data_array[i])
#             # else:
#             #     break
#     finally:
#         connection.close()
#         print("Connection closed")
