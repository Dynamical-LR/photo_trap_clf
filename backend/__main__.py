import asyncio
import base64
from collections import defaultdict
import json
import logging
import os
from typing import AsyncIterable, Optional, cast

import aiofiles
from aiohttp import BodyPartReader, MultipartReader, web
import aiohttp
from aiohttp_session import get_session, new_session, session_middleware
from aiohttp_session.cookie_storage import EncryptedCookieStorage

from cryptography import fernet
from numpy import random
from dino_phototrap import Model

routes = web.RouteTableDef()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

IMAGES_DIR: str = "images"
INDEX_HTML_PATH: str = "index.html"
EXT2CONTENT_TYPE: dict[str, str] = {
    "css": "text/css",
    "js": "text/javascript",
    "png": "image/png",
}

queues = defaultdict(asyncio.Queue)


@routes.get("/predicts")
async def predicts(request: web.Request) -> web.StreamResponse:
    response = web.StreamResponse()
    response.content_type = 'text/plain'
    await response.prepare(request)

    session = await get_session(request)

    while True:
        predict = await queues[session["id"]].get()
        if predict is None:
            break

        b = bytearray(json.dumps({"predict": predict}) + "\n", "utf-8")
        await response.write(b)

    return response


@routes.get("/")
async def index(request: web.Request) -> web.Response:
    """
    Returns index HTML page
    """
    session = await new_session(request)
    session["id"] = random.randint(0, 100000000)

    async with aiofiles.open(INDEX_HTML_PATH) as f:
        html = await f.read()
    return web.Response(text=html, content_type='text/html')


@routes.get(r"/static/{path}")
async def static(request: web.Request) -> web.FileResponse:
    """Returns static files"""
    path: str = request.match_info["path"]
    return web.FileResponse(f"./static/{path}")


@routes.post("/upload-images")
async def upload_images(request: web.Request) -> web.Response:
    """
    Accepts request from `images` HTML form
    Loads all choosed files to `IMAGEX_DIR`
    """
    session = await get_session(request)

    reader: MultipartReader = await request.multipart()
    paths = _download_files(reader)
    print("Files downloaded")

    model = Model("weights/dinov2_5.pth")
    predictions = model(paths)

    print(f"{session.identity=}")

    async for predict in predictions:
        print(f"{predict=}")
        await queues[session["id"]].put(predict)

    await queues[session["id"]].put(None)

    return web.Response(text="OK")


async def _download_files(reader: MultipartReader) -> AsyncIterable[str]:
    """
    Downloads files from `reader` and yields
    paths to files
    """

    field: Optional[BodyPartReader | MultipartReader] = await reader.next()

    while field is not None:
        field = cast(aiohttp.BodyPartReader, field)

        form_name = field.name
        if form_name != 'images':
            raise web.HTTPBadRequest(text="Form should be named 'images'")

        filename = field.filename
        assert filename is not None

        print(f"Downloading {filename=}")
        
        base_path, _ = os.path.split(filename)
        os.makedirs(os.path.join(IMAGES_DIR, base_path), exist_ok=True)

        size = 0
        async with aiofiles.open(os.path.join(IMAGES_DIR, filename), 'wb') as f:
            while True:
                chunk = await field.read_chunk()  # 8192 bytes by default.
                if not chunk:
                    break
                size += len(chunk)
                await f.write(chunk)

        yield os.path.join(IMAGES_DIR, filename)

        field = await reader.next()


def main():
    app = web.Application()

    fernet_key = fernet.Fernet.generate_key()
    secret_key = base64.urlsafe_b64decode(fernet_key)

    app.add_routes(routes)
    app.middlewares.append(session_middleware(EncryptedCookieStorage(secret_key)))

    web.run_app(app)


if __name__ == "__main__":
    main()
