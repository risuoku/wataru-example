import argparse
import importlib

def main():
    parser = argparse.ArgumentParser('entry point')
    parser.add_argument('--name', action='store', dest='scenario_name', default=None)
    ns = parser.parse_args()
    if ns.scenario_name is None:
        raise ValueError('scenario_name must be set!')

    # load scenario module and execute
    mod_scenario = importlib.import_module('modules.scenarios.{}'.format(ns.scenario_name))
    s = mod_scenario.Scenario().build()
    print(s._provider._value)


if __name__ == '__main__':
    main()
