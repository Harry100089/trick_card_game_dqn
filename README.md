After cloning repo, to run app run these commands:  
docker build -t test_ml_project .  
docker run --rm -it test_ml_project  
  
Run with container's shell  
docker run --rm -it test_ml_project /bin/bash  
  
Bind mount. To prevent having to rebuild image after every edit.  
docker run --rm -it -v $(pwd):/app test_ml_project /bin/bash  
  
  
  
Model 1 seems to average 6.7173 out of possible [-20, 20] against random  
Model 2 (upped training eps to 10000 instead of 1000) average 7.56694 in [-20, 20]  
Model 3 (added epsilon decay from 1.0 to 0.01 with 0.995x per episode) average 9.38464 in [-20, 20]