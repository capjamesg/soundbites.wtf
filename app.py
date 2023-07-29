import torch
import os
from flask import Flask, render_template, request, jsonify
import tempfile
import librosa
from sklearn.metrics.pairwise import cosine_similarity
import laion_clap
import json
import datetime

import warnings

# supress UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()

with open("prompts.txt", "r") as f:
    prompts = f.readlines()

date = datetime.datetime.now()

prompt = prompts[date.day % len(prompts)].strip()

embedded_prompt = model.get_text_embedding([prompt, "something else"])

# leaderboard is json, create if not exists
if not os.path.exists("leaderboard.json"):
    with open("leaderboard.json", "w") as f:
        json.dump([], f)

with open("leaderboard.json", "r") as f:
    leaderboard = json.load(f)

app = Flask(__name__, static_folder="./static")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # get body from request audio/ogg; codecs=opus
        audio = request.files["file"]
        # get similarity
        with torch.no_grad():
            # create tmp file
            tmp = tempfile.NamedTemporaryFile()
            audio.save(tmp.name)

            import subprocess

            import random
            import uuid

            file_name = str(uuid.uuid4()) + ".wav"

            subprocess.call(
                [
                    "ffmpeg",
                    "-i",
                    tmp.name,
                    "-acodec",
                    "pcm_s16le",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "./" + file_name,
                ]
            )

            audio_data, _ = librosa.load("./" + file_name)
            audio_data = audio_data.reshape(1, -1) # Make it (1,T) or (N,T)
            audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=False)

            ranking = torch.argsort(torch.tensor(audio_embed) @ torch.tensor(embedded_prompt).t(), descending=True)
            # print percentages
            print(torch.tensor(audio_embed) @ torch.tensor(embedded_prompt).t())

            preds = torch.where(ranking == 0)[1].cpu().numpy()

            # get pred percentagges
            preds = cosine_similarity(audio_embed, embedded_prompt[0].reshape(1, -1))
            preds = torch.tensor(preds)

        # add to leaderboard to the right position depending on similarity
        leaderboard.insert(0, {"username": request.form["username"], "sound": audio.filename, "similarity": preds.tolist()[0][0], "prompt": prompt})
        get_idx = lambda item: item["similarity"]
        leaderboard.sort(key=get_idx, reverse=True)
        # save leaderboard
        with open("leaderboard.json", "w") as f:
            json.dump(leaderboard, f)

        # return similarity
        return jsonify({"similarity": preds.tolist()[0][0]})
    
    # add rank to leaderboard absed on idx
    page_leaderboard = [{"username": item["username"], "sound": item["sound"], "rank": idx + 1} for idx, item in enumerate(leaderboard) if item["prompt"] == prompt]
    
    return render_template("index.html", prompt=prompt, leaderboard=page_leaderboard)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8084, ssl_context="adhoc")