import uvicorn
import argparse

from db.postgres_conn import init_db

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

ROUTER_MODULES = {
    "agent": "agent"
}

argument_parser = argparse.ArgumentParser(
    description='Chat Knowledge Agent', formatter_class=argparse.RawDescriptionHelpFormatter
)
argument_parser.add_argument('-p', '--port', help='Port', metavar='', default="8000")
argument_parser.add_argument('-router', '--router', help='Include Routers', default=None)
argument_parser.add_argument('-worker', '--worker', type=int, default=1)
argument_parser.add_argument('-fenv', '--file_env', default=".env")
args = argument_parser.parse_args()

app = FastAPI(**{
    "title": "Chat Knowledge Agent",
    "description": "REST API for Chat Knowledge Agent"
})


# create tables on startup (dev convenience)
init_db()


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url='/docs')
    # return {"message": "OK"}


for router_name, module_name in ROUTER_MODULES.items():
    if not args.router or (router_name in args.router):
        module = __import__(f"routes.{module_name}", fromlist=["routes"])
        app.include_router(module.router)


if __name__ == '__main__':
    uvicorn.run(
        "main:app", host="0.0.0.0", port=int(args.port), workers=args.worker,
        reload=False if args.worker > 1 else True
    )
