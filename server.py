
import run_experiment
import multiprocessing
import json
from fastapi.templating import Jinja2Templates
import uvicorn
from fastapi import FastAPI, Header, Response, Request, Depends
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse
from starlette.responses import FileResponse 
from fastapi.staticfiles import StaticFiles
import time
import mlflow
app = FastAPI(
    title="ML RL",
    description="ML RL",
    version="1.0",
)
templates = Jinja2Templates(directory="templates")

#app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/") 
def index(request: Request):
  return templates.TemplateResponse("index.html", {"request": request})


@app.post('/handle_data')
async def handle_data(request: Request):
    form = await request.form()
    try :
        cfg = form['op1']
        cfg = json.loads(cfg)
        print(cfg)
        
        #thread = multiprocessing.Process(target= run_experiment.run_experiment,args=(cfg,))
        #thread.start()
        #time.sleep(10)
        experiment_name = 'Type:{} q:{} a:{}'.format( cfg['env']["U_2"],cfg['ddpg']['q']['name'],cfg['ddpg']['a']['name'])
        e = mlflow.get_experiment_by_name(experiment_name)
        df = mlflow.search_runs([e.experiment_id],"attributes.status = 'RUNNING'", order_by=["attribute.start_time DESC"],max_results=1)
        r = df.iloc[0].run_id
        
        
        return templates.TemplateResponse("index.html", {"request": request,"id" : "Deployed experiment at : http://127.0.0.1:5000/#/experiments/{}/runs/{}".format(int(e.experiment_id),r),"json" : cfg})
    
    except Exception as e:
      return templates.TemplateResponse("index.html", {"request": request,"id" : str(e),"json" :cfg})
    #run_experiment(cfg)
    # your code
    # return a response
if __name__ == "__main__":
    uvicorn.run("server:app", port=50000, log_level="debug")