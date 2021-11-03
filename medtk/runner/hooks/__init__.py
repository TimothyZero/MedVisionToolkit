#  Copyright (c) 2020. The Medical Image Computing (MIC) Lab, 陶豪毅
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from .priority import get_priority
from .hook import HOOKS, Hook
from .lr_updater import LrUpdaterHook
from .optimizer import OptimizerHook
from .checkpoint import CheckpointHook
from .iter_timer import IterTimerHook
from .loggers import TextLoggerHook
from .monitor import MonitorHook, ModuleLogHook
from .val_updater import ValUpdaterHook
from .memory import EmptyCacheHook
