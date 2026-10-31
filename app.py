"""Web page for the rule-based check. No model request is sent."""

from flask import Flask, render_template, request

from checker.missing import check_missing

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    result = None
    if request.method == "POST":
        text = request.form.get("description", "")
        result = check_missing(text, "top")
    return render_template("index.html", text=text, result=result, is_empty=not text.strip())


if __name__ == "__main__":
    app.run()
