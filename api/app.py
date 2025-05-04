from flask import Flask, render_template, request, jsonify, make_response
from bot2 import chat

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def indexpage():
    if request.method == "POST":
        print(request.form.get('name'))
        return render_template("index.html")
    return render_template("index.html")

@app.route("/entry", methods=['POST'])
def entry():
    req = request.get_json()
    print(req)
    res = make_response(jsonify({"name": "{}.".format(chat(req)), "message": "OK"}), 200)
    return res

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000) 
