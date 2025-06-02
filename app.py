from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse
import os
from fastapi.staticfiles import StaticFiles
from processing import process_video

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_form():
    with open(os.path.join("static/form.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.post("/upload/", response_class=HTMLResponse)
async def uploadfile(file: UploadFile):
    try:
        file_path = f"{file.filename}"
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        process_video(file_path)

        with open(os.path.join("static/result.html"), "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)


    except Exception as e:
        return {"message": e.args}


