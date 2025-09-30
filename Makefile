

clean:
	rm  -rf  ./aobench/scenario-server/__pycache__
	rm  -rf  ./aobench/scenario-server/src/scenario_server/__pycache__
	rm  -rf  ./src/scenario-server/__pycache__


local:
	uv  sync  --refresh  --reinstall
