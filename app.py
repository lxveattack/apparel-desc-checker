"""Web page. The rule-based result is always computed first, and the model only runs on request."""

from flask import Flask, render_template, request

from checker.llm import analyze
from checker.missing import check_missing

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    result = None
    analysis = None
    if request.method == "POST":
        text = request.form.get("description", "")
        result = check_missing(text, "top")
        if request.form.get("action") == "model":
            analysis = analyze(text, "top", result)
    return render_template("index.html", text=text, result=result, analysis=analysis, is_empty=not text.strip())


if __name__ == "__main__":
    app.run()
