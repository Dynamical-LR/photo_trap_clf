import asyncio
import logging
import os
from typing import cast

import aiofiles
from aiohttp import web
import aiohttp

routes = web.RouteTableDef()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

IMAGES_DIR: str = "images"
INDEX_HTML_PATH: str = "index.html"
EXT2CONTENT_TYPE: dict[str, str] = {
    "css": "text/css",
    "js": "text/javascript"
}

@routes.get("/")
async def index(_: web.Request) -> web.Response:
    """
    Returns index HTML page
    """
    async with aiofiles.open(INDEX_HTML_PATH) as f:
        html = await f.read()
    return web.Response(text=html, content_type='text/html')


@routes.get(r"/static/{path}")
async def static(request: web.Request) -> web.Response:
    """
    Returns static files
    """
    path: str = request.match_info["path"]
    log.info(f"Opening {path} for static")
    async with aiofiles.open(f"static/{path}") as f:
        resource = await f.read()
    ext = path.rsplit(".", 1)[-1]
    return web.Response(text=resource, content_type=EXT2CONTENT_TYPE[ext])


@routes.post("/upload-images")
async def upload_images(request: web.Request) -> web.Response:
    """
    Accepts request from `images` HTML form
    Loads all choosed files to `IMAGEX_DIR`
    """
    # TODO: Add checksum

    tasks = asyncio.tasks.all_tasks()
    log.info(f'TASKS: {len(tasks)}')

    reader = await request.multipart()

    paths = []
    while (field := await reader.next()) is not None:
        field = cast(aiohttp.BodyPartReader, field)

        form_name = field.name
        if form_name != 'images':
            raise web.HTTPBadRequest(text="Form should be named 'images'")

        filename = field.filename
        log.info(f"{form_name=}, {filename=}")
        assert filename is not None
        
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

        paths.append(filename)

    return web.Response(text=f"{len(paths)} files stored")


def main():
    app = web.Application()
    app.add_routes(routes)
    web.run_app(app)


if __name__ == "__main__":
    main()
