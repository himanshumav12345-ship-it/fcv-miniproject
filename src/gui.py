"""
gui.py - the graphical front end: a small web page served from your own computer.

Run:   python -m src.gui          (then open http://127.0.0.1:5000 - it opens by itself)

How it works, in three steps:
  1. The page (src/gui_page.html) sends the chosen photo to  POST /analyse.
  2. This file decodes it and calls predict.predict_array() - the SAME pipeline that
     was used for training and testing - and turns the result into JSON.
  3. The page draws the stage pictures, the disease label, the confidence and the severity.

Bad input never shows a stack trace. InvalidImageError and NoLeafFoundError become a
short message that says what is wrong and what to do about it.

The server listens on 127.0.0.1 only, so nobody else on the network can reach it.
"""
import argparse
import base64
import threading
import webbrowser
from pathlib import Path

import cv2
from flask import Flask, jsonify, request

from src import config, predict
from src.errors import InvalidImageError, NoLeafFoundError
from src.preprocess import decode_image
from src.refine import draw_lesion_overlay
from src.visualize import paint_mask

PAGE_FILE = Path(__file__).with_name("gui_page.html")
MAX_UPLOAD_BYTES = 15 * 1024 * 1024        # refuse absurdly large uploads


def to_data_uri(bgr_image):
    """Encode an image as text ("data:image/png;base64,...") so it can travel inside JSON."""
    ok, png = cv2.imencode(".png", bgr_image)
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")


def stage_panels(analysis):
    """The pictures shown in the GUI: title, one-line explanation, image. In pipeline order."""
    original = analysis.original
    leaf_only = original.copy()
    leaf_only[analysis.leaf_mask == 0] = 238          # light grey, matches the page
    edges = cv2.cvtColor(analysis.edges, cv2.COLOR_GRAY2BGR)
    outlines = draw_lesion_overlay(original, analysis.leaf_mask, analysis.lesions)
    return [
        ("Your photo",
         "The image as loaded, shrunk to at most 256 pixels on its longest side.",
         original),
        ("After blur and contrast",
         "Gaussian blur removes sensor noise; CLAHE raises local contrast on brightness only.",
         analysis.enhanced),
        ("Leaf found",
         "Pixels whose colour differs from the photo's border (the background) are kept as leaf.",
         leaf_only),
        ("Diseased pixels",
         "Leaf pixels whose colour is far from this leaf's own healthy green, after clean-up.",
         paint_mask(original, analysis.lesion_mask)),
        ("Edges",
         "Canny edges, shown for inspection and used as one texture feature. "
         "They do not change the lesion mask.",
         edges),
        ("Regions measured",
         "Leaf outline in green, each lesion outlined in red with its bounding box in yellow.",
         outlines),
    ]


def severity_bands():
    """The severity scale from config, so the page draws exactly the limits the code uses."""
    bands, lower = [], 0.0
    for upper, name in config.SEVERITY_LEVELS:
        bands.append({"name": name, "from": lower, "to": upper})
        lower = upper
    return bands


def prediction_to_json(result):
    """Turn a predict.Prediction into plain data the page can show."""
    analysis = result.analysis
    return {
        "ok": True,
        "disease": result.disease,
        "confidence": result.confidence,
        "probabilities": result.probabilities,
        "severity_percent": analysis.severity_percent,
        "severity_level": analysis.severity_level,
        "severity_bands": severity_bands(),
        "lesion_count": len(analysis.lesions),
        "warnings": result.warnings,
        "panels": [{"title": title, "caption": caption, "image": to_data_uri(image)}
                   for title, caption, image in stage_panels(analysis)],
    }


def error_response(kind, message, status):
    """One shape for every failure: the page shows `message` and picks advice from `kind`."""
    response = jsonify({"ok": False, "kind": kind, "message": message})
    response.status_code = status
    return response


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    @app.get("/")
    def index():
        return PAGE_FILE.read_text(encoding="utf-8")

    @app.get("/favicon.ico")
    def favicon():
        return "", 204                            # no icon; stops the browser's 404 noise

    @app.post("/analyse")
    def analyse():
        upload = request.files.get("photo")
        if upload is None:
            return error_response("no_file", "No photo was sent with the request.", 400)
        try:
            image = decode_image(upload.read())
            result = predict.predict_array(image)
        except InvalidImageError as error:
            return error_response("invalid_image", str(error), 422)
        except NoLeafFoundError as error:
            return error_response("no_leaf", str(error), 422)
        except FileNotFoundError as error:
            return error_response("model_missing", str(error), 503)
        except Exception:                         # a bug of ours: log it, show something calm
            app.logger.exception("analysis failed")
            return error_response("unexpected", "Something went wrong while analysing "
                                  "this photo. Try another photo.", 500)
        return jsonify(prediction_to_json(result))

    @app.errorhandler(413)
    def too_large(_error):
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        return error_response("too_large", f"The file is larger than {limit_mb} MB.", 413)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = parser.parse_args()

    try:
        predict.load_model()                      # load now so a missing model is reported at once
    except FileNotFoundError as error:
        raise SystemExit(str(error))

    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=[url]).start()
    print(f"Leaf disease checker running at {url}   (press Ctrl+C to stop)")
    create_app().run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
