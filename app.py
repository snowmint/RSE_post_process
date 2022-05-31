from flask import Flask, render_template, request, jsonify, send_from_directory
from datetime import datetime
from werkzeug.utils import secure_filename
import mariadb
import sys
import os
import subprocess

app = Flask(__name__, template_folder='templates',
            static_folder='static', static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'static/'
app.config['OUTPUT_FOLDER'] = './static/output/'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # max 20MB
ALLOWED_EXTENSIONS = set(['wav'])


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
        print("./" + app.config['UPLOAD_FOLDER'] + input_filename_no_extension)
        output = subprocess.run(["python", "python_auto_execute.py", "--input",
                                 "./" + app.config['UPLOAD_FOLDER'] + input_filename_no_extension, "--denoise", str(do_denoise), "--output_path", app.config['OUTPUT_FOLDER']], capture_output=True)
        print(output)

        if do_denoise == 1:
            output_filename_no_post = input_filename_no_extension + \
                "_reduction_visualisation_no_post.mid"
            output_filename_final = input_filename_no_extension + \
                "_reduction_visualisation_final.mid"
        else:
            output_filename_no_post = input_filename_no_extension + "_visualisation_no_post.mid"
            output_filename_final = input_filename_no_extension + "_visualisation_final.mid"

        msg = '成功上傳並轉換 ' + input_filename
    else:
        # Invalid file extension type, uplaod only wav files
        msg = '檔案格式錯誤，請上傳 .wav 檔案'

    return jsonify({'htmlresponse': render_template('response.html', msg=msg, input_filename=input_filename,
                                                    output1=output_filename_no_post, output2=output_filename_final)})


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="0.0.0.0", debug=True)
