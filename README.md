<h1 align="center">Welcome to rl-fn 👋</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-blue.svg?cacheSeconds=2592000" />
  <img src="https://img.shields.io/badge/npm-%3E%3D5.5.0-blue.svg" />
  <img src="https://img.shields.io/badge/node-%3E%3D9.3.0-blue.svg" />
  <a href="https://github.com/kefranabg/readme-md-generator#readme" target="_blank">
    <img alt="Documentation" src="https://img.shields.io/badge/documentation-yes-brightgreen.svg" />
  </a>
  <a href="https://github.com/kefranabg/readme-md-generator/graphs/commit-activity" target="_blank">
    <img alt="Maintenance" src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" />
  </a>
  <a href="https://github.com/kefranabg/readme-md-generator/blob/master/LICENSE" target="_blank">
    <img alt="License: MIT" src="https://img.shields.io/github/license/ramsundarg/rl-fn" />
  </a>
</p>

> A DDPG based solution for portfolio optimization that uses specific function classes for actor and critic parts of DDPG instead of neural networks. There are also customized implementations of DDPG (DDPGShockBufferand DDPGShockBufferEstimate)  that improve accuracy as they build better estimates of the expected Q values  of future states. This is a configurable framework where any combination of Q and A can be plugged and played (defined in the folders Qn and An). The environment also can be redefined or configured by editing the BSAvgState file. There is a tunable configuration to experiment with different classes of experiments. Finally there are utilites to plot, visualize the results

### 🏠 [Homepage](https://github.com/ramsundarg/rl-fn)

## Prerequisites

Please check requirements.txt

## Install

```sh
conda create --name <env> --file requirements.txt
```

## Usage

```sh
First create a configuration  for your experiement. Some samples are provided in cfg directory.  Then use the following command.
python run_all.py cfg\EnvSearchAll.json #To run the experiement (along with tunable hyperparameters mentioned in it)

```

## Author

👤 **Ramsundar Govindarajan**

* GitHub: [@ramsundarg](https://github.com/ramsundarg)
* LinkedIn: [@ramsundar-govindarajan-358b126](https://linkedin.com/in/ramsundar-govindarajan-358b126)

## 🤝 Contributing

Contributions, issues and feature requests are welcome!<br />Feel free to check [issues page](https://github.com/kefranabg/readme-md-generator/issues). You can also take a look at the [contributing guide](https://github.com/kefranabg/readme-md-generator/blob/master/CONTRIBUTING.md).

## Show your support

Give a ⭐️ if this project helped you!

## 📝 License

Copyright © 2023 [Ramsundar Govindarajan](https://github.com/ramsundarg).<br />
