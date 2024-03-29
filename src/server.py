
import hypertune
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
  """
    The landing page of the fast API
  """
  return templates.TemplateResponse("index.html", {"request": request})


@app.post('/handle_data')
async def handle_data(request: Request):
    """
      This method executes when the user clicks the submit button to spawn an experiment with all the list of hyperparameter configurations.
      A new experiment will be launched and the mlflow page of the experiment will be returned back. Please note that sometimes the returned URL is unstable and that we need to inspect the MLdashboard separately. Also, mlflow ui should be running on the host where the experiment is run.

      Parameters:
      
        Request: The fastapi Request that contains the cfg of the experiment.

    """
    form = await request.form()
    try :
        cfg = form['op1']
        cfg = json.loads(cfg)
        print(cfg)
        
        thread = multiprocessing.Process(target= hypertune.server_hypertune,args=(cfg,))
        thread.start()
        time.sleep(10)
        experiment_name = '{}Type:{} q:{} a:{}'.format( cfg['name'],cfg['env']["U_2"],cfg['ddpg']['q']['name'],cfg['ddpg']['a']['name'])
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