#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on 7월 08, 2025, at 19:30
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'SC-IAT'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1536, 864]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\이윤주\\Desktop\\Psychopy_IAT\\SC-IAT_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp_intro') is None:
        # initialise key_resp_intro
        key_resp_intro = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_intro',
        )
    if deviceManager.getDevice('key_resp_1') is None:
        # initialise key_resp_1
        key_resp_1 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_1',
        )
    if deviceManager.getDevice('key_resp_intro_2') is None:
        # initialise key_resp_intro_2
        key_resp_intro_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_intro_2',
        )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    if deviceManager.getDevice('key_resp_intro_3') is None:
        # initialise key_resp_intro_3
        key_resp_intro_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_intro_3',
        )
    if deviceManager.getDevice('key_resp_3') is None:
        # initialise key_resp_3
        key_resp_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_3',
        )
    if deviceManager.getDevice('key_resp_intro_4') is None:
        # initialise key_resp_intro_4
        key_resp_intro_4 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_intro_4',
        )
    if deviceManager.getDevice('key_resp_4') is None:
        # initialise key_resp_4
        key_resp_4 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_4',
        )
    if deviceManager.getDevice('End_') is None:
        # initialise End_
        End_ = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='End_',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Intro" ---
    IntroText = visual.TextStim(win=win, name='IntroText',
        text="Z 키에는 왼쪽 손가락을, / 키에는 오른쪽 손가락을 올려주세요. \n화면의 왼쪽 상단에는 '긍정'이, 오른쪽 상단에는 '부정'이 나타납니다.\n중간에 나오는 단어는 다음 세 가지 범주 중 하나에 속할 수 있습니다. \n\n(1) 긍정적인 단어\n(2) 부정적인 단어\n(3) 마약 사진\n\n긍정적인 단어/마약 사진이 나오면 Z 키를 누르고, 부정적인 단어가 나오면 / 키를 눌러주세요.\n\n단어 또는 사진은 한 번에 하나씩 나타나며, 잘못 분류하면 X가 나타납니다.\n정확하게, 가능한 한 빨리 눌러주세요. 준비가 되었다면, 스페이스바를 눌러 시작하세요.  ",
        font='NanumGothic',
        pos=(0, -0.1), draggable=False, height=0.04, wrapWidth=1.7, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Positive_1_1 = visual.TextStim(win=win, name='Positive_1_1',
        text='긍정',
        font='NanumGothic',
        pos=(-0.55, 0.35), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='lightgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    Negative_1_1 = visual.TextStim(win=win, name='Negative_1_1',
        text='부정',
        font='NanumGothic',
        pos=(0.55, 0.35), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='lightgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_intro = keyboard.Keyboard(deviceName='key_resp_intro')
    # Run 'Begin Experiment' code from code_1
    import random
    
    # 자극 리스트
    # 이미지 파일 정의
    positive_images = [
        "기쁘다.jpg", "만족하다.jpg", "흐뭇하다.jpg", "영광스럽다.jpg", "평온하다.jpg",
        "뿌듯하다.jpg", "사랑스럽다.jpg", "반갑다.jpg", "평화롭다.jpg", "상쾌하다.jpg",
        "신나다.jpg", "애정하다.jpg", "유쾌하다.jpg", "재미있다.jpg", "편안하다.jpg",
        "행복하다.jpg", "즐겁다.jpg", "환희하다.jpg", "흥겹다.jpg", "흥미롭다.jpg", "자랑스럽다.jpg"
    ]
    
    negative_images = [
        "거북하다.jpg", "격분하다.jpg", "경멸하다.jpg", "끔찍하다.jpg", "괴롭다.jpg",
        "분노하다.jpg", "불만족하다.jpg", "불쾌하다.jpg", "불행하다.jpg", "비참하다.jpg",
        "섬뜩하다.jpg", "슬프다.jpg", "암담하다.jpg", "역겹다.jpg", "화나다.jpg",
        "실망하다.jpg", "억울하다.jpg", "좌절하다.jpg", "참담하다.jpg", "증오하다.jpg", "질색하다.jpg"
    ]
    
    drug_images = ["drug1.jpg", "drug2.jpg", "drug3.jpg", "drug4.jpg", "drug5.jpg", "drug6.jpg", "drug7.jpg"]
    
    
    # trial 설정
    total_trials = 24
    z_ratio = 0.58
    z_count = round(total_trials * z_ratio)   # 14
    slash_count = total_trials - z_count      # 10
    positive_z = 7
    drug_z = z_count - positive_z  # 7
    
    # 비복원추출 함수
    def strict_sample(images, n):
        if len(images) < n:
            raise ValueError(f"자극이 부족합니다: {len(images)}개 중 {n}개를 요청했습니다.")
        return random.sample(images, n)
    
    # 샘플링
    positive_sample = strict_sample(positive_images, positive_z)
    drug_sample = strict_sample(drug_images, drug_z)
    negative_sample = strict_sample(negative_images, slash_count)
    
    # 자극 풀 구성
    stimuli_pool = (
        [(img, '긍정', 'z') for img in positive_sample] +
        [(img, '마약', 'z') for img in drug_sample] +
        [(img, '부정', '/') for img in negative_sample]
    )
    
    # 무작위 섞기 (중복 방지)
    def is_valid_sequence(seq):
        for i in range(1, len(seq)):
            if seq[i][0] == seq[i-1][0]:
                return False
        return True
    
    for _ in range(1000):
        random.shuffle(stimuli_pool)
        if is_valid_sequence(stimuli_pool):
            break
    else:
        raise RuntimeError("유효한 자극 배열을 생성하지 못했습니다. 이미지 수를 늘려보세요.")
    
    # --- Initialize components for Routine "trial_1" ---
    image_stim_1 = visual.ImageStim(
        win=win,
        name='image_stim_1', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    Drug_1 = visual.TextStim(win=win, name='Drug_1',
        text='마약',
        font='NanumGothic',
        pos=(-0.55, 0.25), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    Positive_1_2 = visual.TextStim(win=win, name='Positive_1_2',
        text='긍정',
        font='NanumGothic',
        pos=(-0.55, 0.35), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='lightgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    Negative_1_2 = visual.TextStim(win=win, name='Negative_1_2',
        text='부정',
        font='NanumGothic',
        pos=(0.55, 0.35), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='lightgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    key_resp_1 = keyboard.Keyboard(deviceName='key_resp_1')
    # Run 'Begin Experiment' code from code_2
    from collections import Counter
    key_counts = Counter([stim[2] for stim in stimuli_pool])
    print(f"✅ trial_1 시작 - stimuli_pool 상태: z: {key_counts['z']}, /: {key_counts['/']}")
    
    # --- Initialize components for Routine "feedback" ---
    msg_feedback_1 = visual.TextStim(win=win, name='msg_feedback_1',
        text='',
        font='Nanumgothic',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "Intro_2" ---
    Positive_2_1 = visual.TextStim(win=win, name='Positive_2_1',
        text='긍정',
        font='NanumGothic',
        pos=(-0.55, 0.35), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='lightgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Negative_2_1 = visual.TextStim(win=win, name='Negative_2_1',
        text='부정',
        font='NanumGothic',
        pos=(0.55, 0.35), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='lightgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_intro_2 = keyboard.Keyboard(deviceName='key_resp_intro_2')
    IntroText_2 = visual.TextStim(win=win, name='IntroText_2',
        text="Z 키에는 왼쪽 손가락을, / 키에는 오른쪽 손가락을 올려주세요. \n화면의 왼쪽 상단에는 '긍정'이, 오른쪽 상단에는 '부정'이 나타납니다.\n중간에 나오는 단어는 다음 세 가지 범주 중 하나에 속할 수 있습니다. \n\n(1) 긍정적인 단어\n(2) 부정적인 단어\n(3) 마약 사진\n\n긍정적인 단어/마약 사진이 나오면 Z 키를 누르고, 부정적인 단어가 나오면 / 키를 눌러주세요.\n\n단어 또는 사진은 한 번에 하나씩 나타나며, 잘못 분류하면 X가 나타납니다.\n정확하게, 가능한 한 빨리 눌러주세요. 준비가 되었다면, 스페이스바를 눌러 시작하세요.  ",
        font='NanumGothic',
        pos=(0, -0.1), draggable=False, height=0.04, wrapWidth=1.7, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "main_loop_" ---
    
    # --- Initialize components for Routine "trial_2" ---
    image_stim_2 = visual.ImageStim(
        win=win,
        name='image_stim_2', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    Drug_2 = visual.TextStim(win=win, name='Drug_2',
        text='마약',
        font='NanumGothic',
        pos=(-0.55, 0.25), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    Positive_2_2 = visual.TextStim(win=win, name='Positive_2_2',
        text='긍정',
        font='NanumGothic',
        pos=(-0.55, 0.35), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='lightgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    Negative_2_2 = visual.TextStim(win=win, name='Negative_2_2',
        text='부정',
        font='NanumGothic',
        pos=(0.55, 0.35), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='lightgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    
    # --- Initialize components for Routine "feedback_2" ---
    msg_feedback_2 = visual.TextStim(win=win, name='msg_feedback_2',
        text='',
        font='Nanumgothic',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "Intro_3" ---
    IntroText_3 = visual.TextStim(win=win, name='IntroText_3',
        text="Z 키에는 왼쪽 손가락을, / 키에는 오른쪽 손가락을 올려주세요. \n화면의 왼쪽 상단에는 '긍정'이, 오른쪽 상단에는 '부정'이 나타납니다.\n중간에 나오는 단어는 다음 세 가지 범주 중 하나에 속할 수 있습니다. \n\n(1) 긍정적인 단어\n(2) 부정적인 단어\n(3) 마약 사진\n\n긍정적인 단어 사진이 나오면 Z 키를 누르고, 부정적인 단어/마약 사진이 나오면 / 키를 눌러주세요.\n\n단어 또는 사진은 한 번에 하나씩 나타나며, 잘못 분류하면 X가 나타납니다.\n정확하게, 가능한 한 빨리 눌러주세요. 준비가 되었다면, 스페이스바를 눌러 시작하세요.  ",
        font='NanumGothic',
        pos=(0, -0.1), draggable=False, height=0.04, wrapWidth=1.7, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Positive_3_1 = visual.TextStim(win=win, name='Positive_3_1',
        text='긍정',
        font='NanumGothic',
        pos=(-0.55, 0.35), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='lightgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    Negative_3_1 = visual.TextStim(win=win, name='Negative_3_1',
        text='부정',
        font='NanumGothic',
        pos=(0.55, 0.35), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='lightgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_intro_3 = keyboard.Keyboard(deviceName='key_resp_intro_3')
    # Run 'Begin Experiment' code from code_8
    import random
    
    # 자극 리스트
    # 이미지 파일 정의
    positive_images = [
        "기쁘다.jpg", "만족하다.jpg", "흐뭇하다.jpg", "영광스럽다.jpg", "평온하다.jpg",
        "뿌듯하다.jpg", "사랑스럽다.jpg", "반갑다.jpg", "평화롭다.jpg", "상쾌하다.jpg",
        "신나다.jpg", "애정하다.jpg", "유쾌하다.jpg", "재미있다.jpg", "편안하다.jpg",
        "행복하다.jpg", "즐겁다.jpg", "환희하다.jpg", "흥겹다.jpg", "흥미롭다.jpg", "자랑스럽다.jpg"
    ]
    
    negative_images = [
        "거북하다.jpg", "격분하다.jpg", "경멸하다.jpg", "끔찍하다.jpg", "괴롭다.jpg",
        "분노하다.jpg", "불만족하다.jpg", "불쾌하다.jpg", "불행하다.jpg", "비참하다.jpg",
        "섬뜩하다.jpg", "슬프다.jpg", "암담하다.jpg", "역겹다.jpg", "화나다.jpg",
        "실망하다.jpg", "억울하다.jpg", "좌절하다.jpg", "참담하다.jpg", "증오하다.jpg", "질색하다.jpg"
    ]
    
    drug_images = ["drug1.jpg", "drug2.jpg", "drug3.jpg", "drug4.jpg", "drug5.jpg", "drug6.jpg", "drug7.jpg"]
    
    
    # trial 설정
    total_trials = 24
    z_ratio = 0.42
    z_count = round(total_trials * z_ratio)   
    slash_count = total_trials - z_count     
    positive_z = 10
    negative_slash = 7
    drug_slash = slash_count - negative_slash  # 7
    
    # 비복원추출 함수
    def strict_sample(images, n):
        if len(images) < n:
            raise ValueError(f"자극이 부족합니다: {len(images)}개 중 {n}개를 요청했습니다.")
        return random.sample(images, n)
    
    # 샘플링
    positive_sample = strict_sample(positive_images, positive_z)
    drug_sample = strict_sample(drug_images, drug_slash)
    negative_sample = strict_sample(negative_images, negative_slash)
    
    # 자극 풀 구성
    stimuli_pool_2 = (
        [(img, '긍정', 'z') for img in positive_sample] +
        [(img, '마약', '/') for img in drug_sample] +
        [(img, '부정', '/') for img in negative_sample]
    )
    
    # 무작위 섞기 (중복 방지)
    def is_valid_sequence(seq):
        for i in range(1, len(seq)):
            if seq[i][0] == seq[i-1][0]:
                return False
        return True
    
    for _ in range(1000):
        random.shuffle(stimuli_pool_2)
        if is_valid_sequence(stimuli_pool_2):
            break
    else:
        raise RuntimeError("유효한 자극 배열을 생성하지 못했습니다. 이미지 수를 늘려보세요.")
    
    # --- Initialize components for Routine "trial_3" ---
    image_stim_3 = visual.ImageStim(
        win=win,
        name='image_stim_3', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    Drug_3 = visual.TextStim(win=win, name='Drug_3',
        text='마약',
        font='NanumGothic',
        pos=(0.55, 0.25), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    Positive_3_2 = visual.TextStim(win=win, name='Positive_3_2',
        text='긍정',
        font='NanumGothic',
        pos=(-0.55, 0.35), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='lightgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    Negative_3_2 = visual.TextStim(win=win, name='Negative_3_2',
        text='부정',
        font='NanumGothic',
        pos=(0.55, 0.35), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='lightgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    key_resp_3 = keyboard.Keyboard(deviceName='key_resp_3')
    
    # --- Initialize components for Routine "feedback_3" ---
    msg_feedback = visual.TextStim(win=win, name='msg_feedback',
        text='',
        font='Nanumgothic',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "Intro_4" ---
    Positive_4_1 = visual.TextStim(win=win, name='Positive_4_1',
        text='긍정',
        font='NanumGothic',
        pos=(-0.55, 0.35), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='lightgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Negative_4_1 = visual.TextStim(win=win, name='Negative_4_1',
        text='부정',
        font='NanumGothic',
        pos=(0.55, 0.35), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='lightgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_intro_4 = keyboard.Keyboard(deviceName='key_resp_intro_4')
    IntroText_4 = visual.TextStim(win=win, name='IntroText_4',
        text="Z 키에는 왼쪽 손가락을, / 키에는 오른쪽 손가락을 올려주세요. \n화면의 왼쪽 상단에는 '긍정'이, 오른쪽 상단에는 '부정'이 나타납니다.\n중간에 나오는 단어는 다음 세 가지 범주 중 하나에 속할 수 있습니다. \n\n(1) 긍정적인 단어\n(2) 부정적인 단어\n(3) 마약 사진\n\n긍정적인 단어 사진이 나오면 Z 키를 누르고, 부정적인 단어/마약 사진이 나오면 / 키를 눌러주세요.\n\n단어 또는 사진은 한 번에 하나씩 나타나며, 잘못 분류하면 X가 나타납니다.\n정확하게, 가능한 한 빨리 눌러주세요. 준비가 되었다면, 스페이스바를 눌러 시작하세요.  ",
        font='NanumGothic',
        pos=(0, -0.1), draggable=False, height=0.04, wrapWidth=1.7, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "main_loop_2" ---
    
    # --- Initialize components for Routine "trial_4" ---
    image_stim_4 = visual.ImageStim(
        win=win,
        name='image_stim_4', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    Drug_4 = visual.TextStim(win=win, name='Drug_4',
        text='마약',
        font='NanumGothic',
        pos=(0.55, 0.25), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    Positive_4_2 = visual.TextStim(win=win, name='Positive_4_2',
        text='긍정',
        font='NanumGothic',
        pos=(-0.55, 0.35), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='lightgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    Negative_4_2 = visual.TextStim(win=win, name='Negative_4_2',
        text='부정',
        font='NanumGothic',
        pos=(0.55, 0.35), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='lightgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    key_resp_4 = keyboard.Keyboard(deviceName='key_resp_4')
    
    # --- Initialize components for Routine "feedback_4" ---
    msg_feedback_3 = visual.TextStim(win=win, name='msg_feedback_3',
        text='',
        font='Nanumgothic',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "End" ---
    text = visual.TextStim(win=win, name='text',
        text='수고하셨습니다!\n참여해주셔서 감사합니다.',
        font='NanumGothic',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    End_ = keyboard.Keyboard(deviceName='End_')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Intro" ---
    # create an object to store info about Routine Intro
    Intro = data.Routine(
        name='Intro',
        components=[IntroText, Positive_1_1, Negative_1_1, key_resp_intro],
    )
    Intro.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_intro
    key_resp_intro.keys = []
    key_resp_intro.rt = []
    _key_resp_intro_allKeys = []
    # Run 'Begin Routine' code from code_1
    message = ''
    fb_color = 'white'
    fb_duration = 0.0
    fb_size = 0.06
    # store start times for Intro
    Intro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Intro.tStart = globalClock.getTime(format='float')
    Intro.status = STARTED
    thisExp.addData('Intro.started', Intro.tStart)
    Intro.maxDuration = None
    # keep track of which components have finished
    IntroComponents = Intro.components
    for thisComponent in Intro.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Intro" ---
    Intro.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *IntroText* updates
        
        # if IntroText is starting this frame...
        if IntroText.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            IntroText.frameNStart = frameN  # exact frame index
            IntroText.tStart = t  # local t and not account for scr refresh
            IntroText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(IntroText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'IntroText.started')
            # update status
            IntroText.status = STARTED
            IntroText.setAutoDraw(True)
        
        # if IntroText is active this frame...
        if IntroText.status == STARTED:
            # update params
            pass
        
        # *Positive_1_1* updates
        
        # if Positive_1_1 is starting this frame...
        if Positive_1_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Positive_1_1.frameNStart = frameN  # exact frame index
            Positive_1_1.tStart = t  # local t and not account for scr refresh
            Positive_1_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Positive_1_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Positive_1_1.started')
            # update status
            Positive_1_1.status = STARTED
            Positive_1_1.setAutoDraw(True)
        
        # if Positive_1_1 is active this frame...
        if Positive_1_1.status == STARTED:
            # update params
            pass
        
        # *Negative_1_1* updates
        
        # if Negative_1_1 is starting this frame...
        if Negative_1_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Negative_1_1.frameNStart = frameN  # exact frame index
            Negative_1_1.tStart = t  # local t and not account for scr refresh
            Negative_1_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Negative_1_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Negative_1_1.started')
            # update status
            Negative_1_1.status = STARTED
            Negative_1_1.setAutoDraw(True)
        
        # if Negative_1_1 is active this frame...
        if Negative_1_1.status == STARTED:
            # update params
            pass
        
        # *key_resp_intro* updates
        waitOnFlip = False
        
        # if key_resp_intro is starting this frame...
        if key_resp_intro.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_intro.frameNStart = frameN  # exact frame index
            key_resp_intro.tStart = t  # local t and not account for scr refresh
            key_resp_intro.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_intro, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_intro.started')
            # update status
            key_resp_intro.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_intro.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_intro.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_intro.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_intro.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_intro_allKeys.extend(theseKeys)
            if len(_key_resp_intro_allKeys):
                key_resp_intro.keys = _key_resp_intro_allKeys[-1].name  # just the last key pressed
                key_resp_intro.rt = _key_resp_intro_allKeys[-1].rt
                key_resp_intro.duration = _key_resp_intro_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Intro.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Intro.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Intro" ---
    for thisComponent in Intro.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Intro
    Intro.tStop = globalClock.getTime(format='float')
    Intro.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Intro.stopped', Intro.tStop)
    # check responses
    if key_resp_intro.keys in ['', [], None]:  # No response was made
        key_resp_intro.keys = None
    thisExp.addData('key_resp_intro.keys',key_resp_intro.keys)
    if key_resp_intro.keys != None:  # we had a response
        thisExp.addData('key_resp_intro.rt', key_resp_intro.rt)
        thisExp.addData('key_resp_intro.duration', key_resp_intro.duration)
    thisExp.nextEntry()
    # the Routine "Intro" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=2.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('stimuli.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "trial_1" ---
        # create an object to store info about Routine trial_1
        trial_1 = data.Routine(
            name='trial_1',
            components=[image_stim_1, Drug_1, Positive_1_2, Negative_1_2, key_resp_1],
        )
        trial_1.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_resp_1
        key_resp_1.keys = []
        key_resp_1.rt = []
        _key_resp_1_allKeys = []
        # Run 'Begin Routine' code from code_2
        # 반드시 맨 위에 global 선언
        global stim_index_1
        
        # 루프의 첫 번째 trial일 때 stim_index 초기화
        if trials.thisN == 0:
            stim_index_1 = 0
        
        # 자극 끝났으면 루틴 종료
        if stim_index_1 >= len(stimuli_pool):
            continueRoutine = False
        else:
            stim_row = stimuli_pool[stim_index_1]
            stimulus_file = stim_row[0]  # 자극 파일명
            stim_category = stim_row[1]  # 자극 범주
            stim_index_1 += 1
            image_stim_1.setImage('images/' + stimulus_file)
        
        # 자극 수 제한 (혹시 모를 예외를 위해 이건 아래에 두는 게 안정적)
        if trials.thisN >= 24:
            continueRoutine = False
        
        # store start times for trial_1
        trial_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial_1.tStart = globalClock.getTime(format='float')
        trial_1.status = STARTED
        thisExp.addData('trial_1.started', trial_1.tStart)
        trial_1.maxDuration = None
        # keep track of which components have finished
        trial_1Components = trial_1.components
        for thisComponent in trial_1.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial_1" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        trial_1.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_stim_1* updates
            
            # if image_stim_1 is starting this frame...
            if image_stim_1.status == NOT_STARTED and tThisFlip >= 0.05-frameTolerance:
                # keep track of start time/frame for later
                image_stim_1.frameNStart = frameN  # exact frame index
                image_stim_1.tStart = t  # local t and not account for scr refresh
                image_stim_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_stim_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_stim_1.started')
                # update status
                image_stim_1.status = STARTED
                image_stim_1.setAutoDraw(True)
            
            # if image_stim_1 is active this frame...
            if image_stim_1.status == STARTED:
                # update params
                pass
            
            # *Drug_1* updates
            
            # if Drug_1 is starting this frame...
            if Drug_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Drug_1.frameNStart = frameN  # exact frame index
                Drug_1.tStart = t  # local t and not account for scr refresh
                Drug_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Drug_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Drug_1.started')
                # update status
                Drug_1.status = STARTED
                Drug_1.setAutoDraw(True)
            
            # if Drug_1 is active this frame...
            if Drug_1.status == STARTED:
                # update params
                pass
            
            # *Positive_1_2* updates
            
            # if Positive_1_2 is starting this frame...
            if Positive_1_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Positive_1_2.frameNStart = frameN  # exact frame index
                Positive_1_2.tStart = t  # local t and not account for scr refresh
                Positive_1_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Positive_1_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Positive_1_2.started')
                # update status
                Positive_1_2.status = STARTED
                Positive_1_2.setAutoDraw(True)
            
            # if Positive_1_2 is active this frame...
            if Positive_1_2.status == STARTED:
                # update params
                pass
            
            # *Negative_1_2* updates
            
            # if Negative_1_2 is starting this frame...
            if Negative_1_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Negative_1_2.frameNStart = frameN  # exact frame index
                Negative_1_2.tStart = t  # local t and not account for scr refresh
                Negative_1_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Negative_1_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Negative_1_2.started')
                # update status
                Negative_1_2.status = STARTED
                Negative_1_2.setAutoDraw(True)
            
            # if Negative_1_2 is active this frame...
            if Negative_1_2.status == STARTED:
                # update params
                pass
            
            # *key_resp_1* updates
            waitOnFlip = False
            
            # if key_resp_1 is starting this frame...
            if key_resp_1.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                # keep track of start time/frame for later
                key_resp_1.frameNStart = frameN  # exact frame index
                key_resp_1.tStart = t  # local t and not account for scr refresh
                key_resp_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_1.started')
                # update status
                key_resp_1.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_1.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_1.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_1.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_1.getKeys(keyList=['z','slash'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_1_allKeys.extend(theseKeys)
                if len(_key_resp_1_allKeys):
                    key_resp_1.keys = _key_resp_1_allKeys[-1].name  # just the last key pressed
                    key_resp_1.rt = _key_resp_1_allKeys[-1].rt
                    key_resp_1.duration = _key_resp_1_allKeys[-1].duration
                    # was this correct?
                    if (key_resp_1.keys == str(Corr)) or (key_resp_1.keys == Corr):
                        key_resp_1.corr = 1
                    else:
                        key_resp_1.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trial_1.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial_1.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_1" ---
        for thisComponent in trial_1.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial_1
        trial_1.tStop = globalClock.getTime(format='float')
        trial_1.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial_1.stopped', trial_1.tStop)
        # check responses
        if key_resp_1.keys in ['', [], None]:  # No response was made
            key_resp_1.keys = None
            # was no response the correct answer?!
            if str(Corr).lower() == 'none':
               key_resp_1.corr = 1;  # correct non-response
            else:
               key_resp_1.corr = 0;  # failed to respond (incorrectly)
        # store data for trials (TrialHandler)
        trials.addData('key_resp_1.keys',key_resp_1.keys)
        trials.addData('key_resp_1.corr', key_resp_1.corr)
        if key_resp_1.keys != None:  # we had a response
            trials.addData('key_resp_1.rt', key_resp_1.rt)
            trials.addData('key_resp_1.duration', key_resp_1.duration)
        # Run 'End Routine' code from code_2
        # 키 표준화 (슬래시 키 대응)
        if key_resp_1.keys == 'slash':
            key_resp_1.keys = '/'
        
        # 정답 키 설정
        if stim_category == '부정':
            correct_key = '/'
        else:
            correct_key = 'z'
        
        # 채점 처리
        if key_resp_1.keys == correct_key:
            key_resp_1.corr = 1
        else:
            key_resp_1.corr = 0
        
        if stim_index_1 >= 24:
            trials.finished = True  # 루프 종료
        # the Routine "trial_1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "feedback" ---
        # create an object to store info about Routine feedback
        feedback = data.Routine(
            name='feedback',
            components=[msg_feedback_1],
        )
        feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_3
        if key_resp_1.keys:  # 반응이 있었을 경우
            if key_resp_1.rt is not None and key_resp_1.rt > 1.5:
                message = '더 빠르게 해주세요'
                fb_color = 'white'
                fb_duration = 0.5
                fb_size = 0.06  # 작게
            elif key_resp_1.corr == 1:
                message = 'O'
                fb_color = 'green'
                fb_duration = 0.15
                fb_size = 0.1  # 크게
            elif key_resp_1.corr == 0:
                message = 'X'
                fb_color = 'red'
                fb_duration = 0.15
                fb_size = 0.1  # 크게
        
        else:  # 반응이 없었을 때
            message = '더 빠르게 해주세요'
            fb_color = 'white'
            fb_duration = 0.5
            fb_size = 0.06  # 작게
        msg_feedback_1.setColor(fb_color, colorSpace='rgb')
        msg_feedback_1.setText(message)
        msg_feedback_1.setHeight(fb_size)
        # store start times for feedback
        feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        feedback.tStart = globalClock.getTime(format='float')
        feedback.status = STARTED
        thisExp.addData('feedback.started', feedback.tStart)
        feedback.maxDuration = None
        # keep track of which components have finished
        feedbackComponents = feedback.components
        for thisComponent in feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "feedback" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *msg_feedback_1* updates
            
            # if msg_feedback_1 is starting this frame...
            if msg_feedback_1.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                # keep track of start time/frame for later
                msg_feedback_1.frameNStart = frameN  # exact frame index
                msg_feedback_1.tStart = t  # local t and not account for scr refresh
                msg_feedback_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(msg_feedback_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'msg_feedback_1.started')
                # update status
                msg_feedback_1.status = STARTED
                msg_feedback_1.setAutoDraw(True)
            
            # if msg_feedback_1 is active this frame...
            if msg_feedback_1.status == STARTED:
                # update params
                pass
            
            # if msg_feedback_1 is stopping this frame...
            if msg_feedback_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > msg_feedback_1.tStartRefresh + fb_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    msg_feedback_1.tStop = t  # not accounting for scr refresh
                    msg_feedback_1.tStopRefresh = tThisFlipGlobal  # on global time
                    msg_feedback_1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'msg_feedback_1.stopped')
                    # update status
                    msg_feedback_1.status = FINISHED
                    msg_feedback_1.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback" ---
        for thisComponent in feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for feedback
        feedback.tStop = globalClock.getTime(format='float')
        feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('feedback.stopped', feedback.tStop)
        # the Routine "feedback" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 2.0 repeats of 'trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Intro_2" ---
    # create an object to store info about Routine Intro_2
    Intro_2 = data.Routine(
        name='Intro_2',
        components=[Positive_2_1, Negative_2_1, key_resp_intro_2, IntroText_2],
    )
    Intro_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_intro_2
    key_resp_intro_2.keys = []
    key_resp_intro_2.rt = []
    _key_resp_intro_2_allKeys = []
    # Run 'Begin Routine' code from code_4
    message = ''
    fb_color = 'white'
    fb_duration = 0.0
    fb_size = 0.06
    # store start times for Intro_2
    Intro_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Intro_2.tStart = globalClock.getTime(format='float')
    Intro_2.status = STARTED
    thisExp.addData('Intro_2.started', Intro_2.tStart)
    Intro_2.maxDuration = None
    # keep track of which components have finished
    Intro_2Components = Intro_2.components
    for thisComponent in Intro_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Intro_2" ---
    Intro_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Positive_2_1* updates
        
        # if Positive_2_1 is starting this frame...
        if Positive_2_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Positive_2_1.frameNStart = frameN  # exact frame index
            Positive_2_1.tStart = t  # local t and not account for scr refresh
            Positive_2_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Positive_2_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Positive_2_1.started')
            # update status
            Positive_2_1.status = STARTED
            Positive_2_1.setAutoDraw(True)
        
        # if Positive_2_1 is active this frame...
        if Positive_2_1.status == STARTED:
            # update params
            pass
        
        # *Negative_2_1* updates
        
        # if Negative_2_1 is starting this frame...
        if Negative_2_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Negative_2_1.frameNStart = frameN  # exact frame index
            Negative_2_1.tStart = t  # local t and not account for scr refresh
            Negative_2_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Negative_2_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Negative_2_1.started')
            # update status
            Negative_2_1.status = STARTED
            Negative_2_1.setAutoDraw(True)
        
        # if Negative_2_1 is active this frame...
        if Negative_2_1.status == STARTED:
            # update params
            pass
        
        # *key_resp_intro_2* updates
        waitOnFlip = False
        
        # if key_resp_intro_2 is starting this frame...
        if key_resp_intro_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_intro_2.frameNStart = frameN  # exact frame index
            key_resp_intro_2.tStart = t  # local t and not account for scr refresh
            key_resp_intro_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_intro_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_intro_2.started')
            # update status
            key_resp_intro_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_intro_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_intro_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_intro_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_intro_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_intro_2_allKeys.extend(theseKeys)
            if len(_key_resp_intro_2_allKeys):
                key_resp_intro_2.keys = _key_resp_intro_2_allKeys[-1].name  # just the last key pressed
                key_resp_intro_2.rt = _key_resp_intro_2_allKeys[-1].rt
                key_resp_intro_2.duration = _key_resp_intro_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *IntroText_2* updates
        
        # if IntroText_2 is starting this frame...
        if IntroText_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            IntroText_2.frameNStart = frameN  # exact frame index
            IntroText_2.tStart = t  # local t and not account for scr refresh
            IntroText_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(IntroText_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'IntroText_2.started')
            # update status
            IntroText_2.status = STARTED
            IntroText_2.setAutoDraw(True)
        
        # if IntroText_2 is active this frame...
        if IntroText_2.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Intro_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Intro_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Intro_2" ---
    for thisComponent in Intro_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Intro_2
    Intro_2.tStop = globalClock.getTime(format='float')
    Intro_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Intro_2.stopped', Intro_2.tStop)
    # check responses
    if key_resp_intro_2.keys in ['', [], None]:  # No response was made
        key_resp_intro_2.keys = None
    thisExp.addData('key_resp_intro_2.keys',key_resp_intro_2.keys)
    if key_resp_intro_2.keys != None:  # we had a response
        thisExp.addData('key_resp_intro_2.rt', key_resp_intro_2.rt)
        thisExp.addData('key_resp_intro_2.duration', key_resp_intro_2.duration)
    thisExp.nextEntry()
    # the Routine "Intro_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    main_loop = data.TrialHandler2(
        name='main_loop',
        nReps=3.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(main_loop)  # add the loop to the experiment
    thisMain_loop = main_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisMain_loop.rgb)
    if thisMain_loop != None:
        for paramName in thisMain_loop:
            globals()[paramName] = thisMain_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisMain_loop in main_loop:
        currentLoop = main_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisMain_loop.rgb)
        if thisMain_loop != None:
            for paramName in thisMain_loop:
                globals()[paramName] = thisMain_loop[paramName]
        
        # --- Prepare to start Routine "main_loop_" ---
        # create an object to store info about Routine main_loop_
        main_loop_ = data.Routine(
            name='main_loop_',
            components=[],
        )
        main_loop_.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_5
        import pandas as pd
        
        # 전체 자극 목록 불러오기
        df = pd.read_excel('stimuli.xlsx')
        
        # 24개 자극 비복원추출
        subset_df_1 = df.sample(n=24, replace=False).reset_index(drop=True)
        
        # 반복마다 다른 파일 이름으로 저장 (필수는 아님)
        subset_df_1.to_csv('temp_conditions.csv', index=False)
        
        # store start times for main_loop_
        main_loop_.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        main_loop_.tStart = globalClock.getTime(format='float')
        main_loop_.status = STARTED
        thisExp.addData('main_loop_.started', main_loop_.tStart)
        main_loop_.maxDuration = None
        # keep track of which components have finished
        main_loop_Components = main_loop_.components
        for thisComponent in main_loop_.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "main_loop_" ---
        # if trial has changed, end Routine now
        if isinstance(main_loop, data.TrialHandler2) and thisMain_loop.thisN != main_loop.thisTrial.thisN:
            continueRoutine = False
        main_loop_.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                main_loop_.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in main_loop_.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "main_loop_" ---
        for thisComponent in main_loop_.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for main_loop_
        main_loop_.tStop = globalClock.getTime(format='float')
        main_loop_.tStopRefresh = tThisFlipGlobal
        thisExp.addData('main_loop_.stopped', main_loop_.tStop)
        # the Routine "main_loop_" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials_2 = data.TrialHandler2(
            name='trials_2',
            nReps=1.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('temp_conditions.csv'), 
            seed=None, 
        )
        thisExp.addLoop(trials_2)  # add the loop to the experiment
        thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
        if thisTrial_2 != None:
            for paramName in thisTrial_2:
                globals()[paramName] = thisTrial_2[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrial_2 in trials_2:
            currentLoop = trials_2
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
            if thisTrial_2 != None:
                for paramName in thisTrial_2:
                    globals()[paramName] = thisTrial_2[paramName]
            
            # --- Prepare to start Routine "trial_2" ---
            # create an object to store info about Routine trial_2
            trial_2 = data.Routine(
                name='trial_2',
                components=[image_stim_2, Drug_2, Positive_2_2, Negative_2_2, key_resp_2],
            )
            trial_2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for key_resp_2
            key_resp_2.keys = []
            key_resp_2.rt = []
            _key_resp_2_allKeys = []
            # Run 'Begin Routine' code from code_6
            if trials_2.thisN == 0:
                global stim_index_2
                stim_index_2 = 0
            
            if stim_index_2 >= len(stimuli_pool):
                continueRoutine = False
            else:
                stim_row = stimuli_pool[stim_index_2]
                stimulus_file = stim_row[0]       # 파일명
                stim_category = stim_row[1]       # 카테고리
                stim_index_2 += 1
                image_stim_2.setImage('images/' + stimulus_file)
            
            if filename == '' or filename is None:
                continueRoutine = False
            
            
            # store start times for trial_2
            trial_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial_2.tStart = globalClock.getTime(format='float')
            trial_2.status = STARTED
            thisExp.addData('trial_2.started', trial_2.tStart)
            trial_2.maxDuration = None
            # keep track of which components have finished
            trial_2Components = trial_2.components
            for thisComponent in trial_2.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial_2" ---
            # if trial has changed, end Routine now
            if isinstance(trials_2, data.TrialHandler2) and thisTrial_2.thisN != trials_2.thisTrial.thisN:
                continueRoutine = False
            trial_2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image_stim_2* updates
                
                # if image_stim_2 is starting this frame...
                if image_stim_2.status == NOT_STARTED and tThisFlip >= 0.05-frameTolerance:
                    # keep track of start time/frame for later
                    image_stim_2.frameNStart = frameN  # exact frame index
                    image_stim_2.tStart = t  # local t and not account for scr refresh
                    image_stim_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_stim_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_stim_2.started')
                    # update status
                    image_stim_2.status = STARTED
                    image_stim_2.setAutoDraw(True)
                
                # if image_stim_2 is active this frame...
                if image_stim_2.status == STARTED:
                    # update params
                    pass
                
                # *Drug_2* updates
                
                # if Drug_2 is starting this frame...
                if Drug_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    Drug_2.frameNStart = frameN  # exact frame index
                    Drug_2.tStart = t  # local t and not account for scr refresh
                    Drug_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Drug_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Drug_2.started')
                    # update status
                    Drug_2.status = STARTED
                    Drug_2.setAutoDraw(True)
                
                # if Drug_2 is active this frame...
                if Drug_2.status == STARTED:
                    # update params
                    pass
                
                # *Positive_2_2* updates
                
                # if Positive_2_2 is starting this frame...
                if Positive_2_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    Positive_2_2.frameNStart = frameN  # exact frame index
                    Positive_2_2.tStart = t  # local t and not account for scr refresh
                    Positive_2_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Positive_2_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Positive_2_2.started')
                    # update status
                    Positive_2_2.status = STARTED
                    Positive_2_2.setAutoDraw(True)
                
                # if Positive_2_2 is active this frame...
                if Positive_2_2.status == STARTED:
                    # update params
                    pass
                
                # *Negative_2_2* updates
                
                # if Negative_2_2 is starting this frame...
                if Negative_2_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    Negative_2_2.frameNStart = frameN  # exact frame index
                    Negative_2_2.tStart = t  # local t and not account for scr refresh
                    Negative_2_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Negative_2_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Negative_2_2.started')
                    # update status
                    Negative_2_2.status = STARTED
                    Negative_2_2.setAutoDraw(True)
                
                # if Negative_2_2 is active this frame...
                if Negative_2_2.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_2* updates
                waitOnFlip = False
                
                # if key_resp_2 is starting this frame...
                if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_2.frameNStart = frameN  # exact frame index
                    key_resp_2.tStart = t  # local t and not account for scr refresh
                    key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_2.started')
                    # update status
                    key_resp_2.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_2.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_2.getKeys(keyList=['z','slash'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_2_allKeys.extend(theseKeys)
                    if len(_key_resp_2_allKeys):
                        key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                        key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                        key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                        # was this correct?
                        if (key_resp_2.keys == str(Corr)) or (key_resp_2.keys == Corr):
                            key_resp_2.corr = 1
                        else:
                            key_resp_2.corr = 0
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    trial_2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial_2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial_2" ---
            for thisComponent in trial_2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial_2
            trial_2.tStop = globalClock.getTime(format='float')
            trial_2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial_2.stopped', trial_2.tStop)
            # check responses
            if key_resp_2.keys in ['', [], None]:  # No response was made
                key_resp_2.keys = None
                # was no response the correct answer?!
                if str(Corr).lower() == 'none':
                   key_resp_2.corr = 1;  # correct non-response
                else:
                   key_resp_2.corr = 0;  # failed to respond (incorrectly)
            # store data for trials_2 (TrialHandler)
            trials_2.addData('key_resp_2.keys',key_resp_2.keys)
            trials_2.addData('key_resp_2.corr', key_resp_2.corr)
            if key_resp_2.keys != None:  # we had a response
                trials_2.addData('key_resp_2.rt', key_resp_2.rt)
                trials_2.addData('key_resp_2.duration', key_resp_2.duration)
            # Run 'End Routine' code from code_6
            # 키 추출 및 표준화
            keys = key_resp_2.keys
            if isinstance(keys, list):
                keys = keys[0] if keys else None
                
            # 키 표준화 (슬래시 키 대응)
            if key_resp_2.keys == 'slash':
                key_resp_2.keys = '/'
            
            # 정답 키 설정
            if stim_category == '부정':
                correct_key = '/'
            else:
                correct_key = 'z'
            
            # 채점 처리
            if key_resp_2.keys == correct_key:
                key_resp_2.corr = 1
            else:
                key_resp_2.corr = 0
            
            if stim_index_2 >= 24:
                trials.finished = True  # 루프 종료
            # the Routine "trial_2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "feedback_2" ---
            # create an object to store info about Routine feedback_2
            feedback_2 = data.Routine(
                name='feedback_2',
                components=[msg_feedback_2],
            )
            feedback_2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_7
            if trials_2.thisTrial is None:
                continueRoutine = False
                
            if key_resp_2.keys:  # 반응이 있었을 경우
                if key_resp_2.rt is not None and key_resp_2.rt > 1.5:
                    message = '더 빠르게 해주세요'
                    fb_color = 'white'
                    fb_duration = 0.5
                    fb_size = 0.06  # 작게
                elif key_resp_2.corr == 1:
                    message = 'O'
                    fb_color = 'green'
                    fb_duration = 0.15
                    fb_size = 0.1  # 크게
                elif key_resp_2.corr == 0:
                    message = 'X'
                    fb_color = 'red'
                    fb_duration = 0.15
                    fb_size = 0.1  # 크게
            
            else:  # 반응이 없었을 때
                message = '더 빠르게 해주세요'
                fb_color = 'white'
                fb_duration = 0.5
                fb_size = 0.06  # 작게
            
            msg_feedback_2.setColor(fb_color, colorSpace='rgb')
            msg_feedback_2.setText(message)
            msg_feedback_2.setHeight(fb_size)
            # store start times for feedback_2
            feedback_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            feedback_2.tStart = globalClock.getTime(format='float')
            feedback_2.status = STARTED
            thisExp.addData('feedback_2.started', feedback_2.tStart)
            feedback_2.maxDuration = None
            # keep track of which components have finished
            feedback_2Components = feedback_2.components
            for thisComponent in feedback_2.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "feedback_2" ---
            # if trial has changed, end Routine now
            if isinstance(trials_2, data.TrialHandler2) and thisTrial_2.thisN != trials_2.thisTrial.thisN:
                continueRoutine = False
            feedback_2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *msg_feedback_2* updates
                
                # if msg_feedback_2 is starting this frame...
                if msg_feedback_2.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                    # keep track of start time/frame for later
                    msg_feedback_2.frameNStart = frameN  # exact frame index
                    msg_feedback_2.tStart = t  # local t and not account for scr refresh
                    msg_feedback_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(msg_feedback_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'msg_feedback_2.started')
                    # update status
                    msg_feedback_2.status = STARTED
                    msg_feedback_2.setAutoDraw(True)
                
                # if msg_feedback_2 is active this frame...
                if msg_feedback_2.status == STARTED:
                    # update params
                    pass
                
                # if msg_feedback_2 is stopping this frame...
                if msg_feedback_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > msg_feedback_2.tStartRefresh + fb_duration-frameTolerance:
                        # keep track of stop time/frame for later
                        msg_feedback_2.tStop = t  # not accounting for scr refresh
                        msg_feedback_2.tStopRefresh = tThisFlipGlobal  # on global time
                        msg_feedback_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'msg_feedback_2.stopped')
                        # update status
                        msg_feedback_2.status = FINISHED
                        msg_feedback_2.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    feedback_2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in feedback_2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "feedback_2" ---
            for thisComponent in feedback_2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for feedback_2
            feedback_2.tStop = globalClock.getTime(format='float')
            feedback_2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('feedback_2.stopped', feedback_2.tStop)
            # the Routine "feedback_2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'trials_2'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        thisExp.nextEntry()
        
    # completed 3.0 repeats of 'main_loop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Intro_3" ---
    # create an object to store info about Routine Intro_3
    Intro_3 = data.Routine(
        name='Intro_3',
        components=[IntroText_3, Positive_3_1, Negative_3_1, key_resp_intro_3],
    )
    Intro_3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_intro_3
    key_resp_intro_3.keys = []
    key_resp_intro_3.rt = []
    _key_resp_intro_3_allKeys = []
    # Run 'Begin Routine' code from code_8
    message = ''
    fb_color = 'white'
    fb_duration = 0.0
    fb_size = 0.06
    # store start times for Intro_3
    Intro_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Intro_3.tStart = globalClock.getTime(format='float')
    Intro_3.status = STARTED
    thisExp.addData('Intro_3.started', Intro_3.tStart)
    Intro_3.maxDuration = None
    # keep track of which components have finished
    Intro_3Components = Intro_3.components
    for thisComponent in Intro_3.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Intro_3" ---
    Intro_3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *IntroText_3* updates
        
        # if IntroText_3 is starting this frame...
        if IntroText_3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            IntroText_3.frameNStart = frameN  # exact frame index
            IntroText_3.tStart = t  # local t and not account for scr refresh
            IntroText_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(IntroText_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'IntroText_3.started')
            # update status
            IntroText_3.status = STARTED
            IntroText_3.setAutoDraw(True)
        
        # if IntroText_3 is active this frame...
        if IntroText_3.status == STARTED:
            # update params
            pass
        
        # *Positive_3_1* updates
        
        # if Positive_3_1 is starting this frame...
        if Positive_3_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Positive_3_1.frameNStart = frameN  # exact frame index
            Positive_3_1.tStart = t  # local t and not account for scr refresh
            Positive_3_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Positive_3_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Positive_3_1.started')
            # update status
            Positive_3_1.status = STARTED
            Positive_3_1.setAutoDraw(True)
        
        # if Positive_3_1 is active this frame...
        if Positive_3_1.status == STARTED:
            # update params
            pass
        
        # *Negative_3_1* updates
        
        # if Negative_3_1 is starting this frame...
        if Negative_3_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Negative_3_1.frameNStart = frameN  # exact frame index
            Negative_3_1.tStart = t  # local t and not account for scr refresh
            Negative_3_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Negative_3_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Negative_3_1.started')
            # update status
            Negative_3_1.status = STARTED
            Negative_3_1.setAutoDraw(True)
        
        # if Negative_3_1 is active this frame...
        if Negative_3_1.status == STARTED:
            # update params
            pass
        
        # *key_resp_intro_3* updates
        waitOnFlip = False
        
        # if key_resp_intro_3 is starting this frame...
        if key_resp_intro_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_intro_3.frameNStart = frameN  # exact frame index
            key_resp_intro_3.tStart = t  # local t and not account for scr refresh
            key_resp_intro_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_intro_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_intro_3.started')
            # update status
            key_resp_intro_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_intro_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_intro_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_intro_3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_intro_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_intro_3_allKeys.extend(theseKeys)
            if len(_key_resp_intro_3_allKeys):
                key_resp_intro_3.keys = _key_resp_intro_3_allKeys[-1].name  # just the last key pressed
                key_resp_intro_3.rt = _key_resp_intro_3_allKeys[-1].rt
                key_resp_intro_3.duration = _key_resp_intro_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Intro_3.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Intro_3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Intro_3" ---
    for thisComponent in Intro_3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Intro_3
    Intro_3.tStop = globalClock.getTime(format='float')
    Intro_3.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Intro_3.stopped', Intro_3.tStop)
    # check responses
    if key_resp_intro_3.keys in ['', [], None]:  # No response was made
        key_resp_intro_3.keys = None
    thisExp.addData('key_resp_intro_3.keys',key_resp_intro_3.keys)
    if key_resp_intro_3.keys != None:  # we had a response
        thisExp.addData('key_resp_intro_3.rt', key_resp_intro_3.rt)
        thisExp.addData('key_resp_intro_3.duration', key_resp_intro_3.duration)
    thisExp.nextEntry()
    # the Routine "Intro_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_3 = data.TrialHandler2(
        name='trials_3',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('stimuli.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials_3)  # add the loop to the experiment
    thisTrial_3 = trials_3.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_3.rgb)
    if thisTrial_3 != None:
        for paramName in thisTrial_3:
            globals()[paramName] = thisTrial_3[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_3 in trials_3:
        currentLoop = trials_3
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_3.rgb)
        if thisTrial_3 != None:
            for paramName in thisTrial_3:
                globals()[paramName] = thisTrial_3[paramName]
        
        # --- Prepare to start Routine "trial_3" ---
        # create an object to store info about Routine trial_3
        trial_3 = data.Routine(
            name='trial_3',
            components=[image_stim_3, Drug_3, Positive_3_2, Negative_3_2, key_resp_3],
        )
        trial_3.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_resp_3
        key_resp_3.keys = []
        key_resp_3.rt = []
        _key_resp_3_allKeys = []
        # Run 'Begin Routine' code from code_9
        # 반드시 맨 위에 global 선언
        global stim_index_3
        
        # 루프의 첫 번째 trial일 때 stim_index 초기화
        if trials_3.thisN == 0:
            stim_index_3 = 0
        
        # 자극 끝났으면 루틴 종료
        if stim_index_3 >= len(stimuli_pool_2):
            continueRoutine = False
        else:
            stim_row = stimuli_pool_2[stim_index_3]
            stimulus_file = stim_row[0]  # 자극 파일명
            stim_category = stim_row[1]  # 자극 범주
            stim_index_3 += 1
            image_stim_3.setImage('images/' + stimulus_file)
        
        # 자극 수 제한 (혹시 모를 예외를 위해 이건 아래에 두는 게 안정적)
        if trials_3.thisN >= 24:
            continueRoutine = False
        
        # store start times for trial_3
        trial_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial_3.tStart = globalClock.getTime(format='float')
        trial_3.status = STARTED
        thisExp.addData('trial_3.started', trial_3.tStart)
        trial_3.maxDuration = None
        # keep track of which components have finished
        trial_3Components = trial_3.components
        for thisComponent in trial_3.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial_3" ---
        # if trial has changed, end Routine now
        if isinstance(trials_3, data.TrialHandler2) and thisTrial_3.thisN != trials_3.thisTrial.thisN:
            continueRoutine = False
        trial_3.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_stim_3* updates
            
            # if image_stim_3 is starting this frame...
            if image_stim_3.status == NOT_STARTED and tThisFlip >= 0.05-frameTolerance:
                # keep track of start time/frame for later
                image_stim_3.frameNStart = frameN  # exact frame index
                image_stim_3.tStart = t  # local t and not account for scr refresh
                image_stim_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_stim_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_stim_3.started')
                # update status
                image_stim_3.status = STARTED
                image_stim_3.setAutoDraw(True)
            
            # if image_stim_3 is active this frame...
            if image_stim_3.status == STARTED:
                # update params
                pass
            
            # *Drug_3* updates
            
            # if Drug_3 is starting this frame...
            if Drug_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Drug_3.frameNStart = frameN  # exact frame index
                Drug_3.tStart = t  # local t and not account for scr refresh
                Drug_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Drug_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Drug_3.started')
                # update status
                Drug_3.status = STARTED
                Drug_3.setAutoDraw(True)
            
            # if Drug_3 is active this frame...
            if Drug_3.status == STARTED:
                # update params
                pass
            
            # *Positive_3_2* updates
            
            # if Positive_3_2 is starting this frame...
            if Positive_3_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Positive_3_2.frameNStart = frameN  # exact frame index
                Positive_3_2.tStart = t  # local t and not account for scr refresh
                Positive_3_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Positive_3_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Positive_3_2.started')
                # update status
                Positive_3_2.status = STARTED
                Positive_3_2.setAutoDraw(True)
            
            # if Positive_3_2 is active this frame...
            if Positive_3_2.status == STARTED:
                # update params
                pass
            
            # *Negative_3_2* updates
            
            # if Negative_3_2 is starting this frame...
            if Negative_3_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Negative_3_2.frameNStart = frameN  # exact frame index
                Negative_3_2.tStart = t  # local t and not account for scr refresh
                Negative_3_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Negative_3_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Negative_3_2.started')
                # update status
                Negative_3_2.status = STARTED
                Negative_3_2.setAutoDraw(True)
            
            # if Negative_3_2 is active this frame...
            if Negative_3_2.status == STARTED:
                # update params
                pass
            
            # *key_resp_3* updates
            waitOnFlip = False
            
            # if key_resp_3 is starting this frame...
            if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                # keep track of start time/frame for later
                key_resp_3.frameNStart = frameN  # exact frame index
                key_resp_3.tStart = t  # local t and not account for scr refresh
                key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_3.started')
                # update status
                key_resp_3.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_3.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_3.getKeys(keyList=['z','slash'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_3_allKeys.extend(theseKeys)
                if len(_key_resp_3_allKeys):
                    key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                    key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                    key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                    # was this correct?
                    if (key_resp_3.keys == str(Corr)) or (key_resp_3.keys == Corr):
                        key_resp_3.corr = 1
                    else:
                        key_resp_3.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trial_3.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial_3.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_3" ---
        for thisComponent in trial_3.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial_3
        trial_3.tStop = globalClock.getTime(format='float')
        trial_3.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial_3.stopped', trial_3.tStop)
        # check responses
        if key_resp_3.keys in ['', [], None]:  # No response was made
            key_resp_3.keys = None
            # was no response the correct answer?!
            if str(Corr).lower() == 'none':
               key_resp_3.corr = 1;  # correct non-response
            else:
               key_resp_3.corr = 0;  # failed to respond (incorrectly)
        # store data for trials_3 (TrialHandler)
        trials_3.addData('key_resp_3.keys',key_resp_3.keys)
        trials_3.addData('key_resp_3.corr', key_resp_3.corr)
        if key_resp_3.keys != None:  # we had a response
            trials_3.addData('key_resp_3.rt', key_resp_3.rt)
            trials_3.addData('key_resp_3.duration', key_resp_3.duration)
        # Run 'End Routine' code from code_9
        # 키 표준화 (슬래시 키 대응)
        if key_resp_3.keys == 'slash':
            key_resp_3.keys = '/'
        
        # 정답 키 설정
        if stim_category == '긍정':
            correct_key = 'z'
        else:
            correct_key = '/'
        
        # 채점 처리
        if key_resp_3.keys == correct_key:
            key_resp_3.corr = 1
        else:
            key_resp_3.corr = 0
        
        if stim_index_3 >= 24:
            trials_3.finished = True  # 루프 종료
        # the Routine "trial_3" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "feedback_3" ---
        # create an object to store info about Routine feedback_3
        feedback_3 = data.Routine(
            name='feedback_3',
            components=[msg_feedback],
        )
        feedback_3.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_10
        if key_resp_3.keys:  # 반응이 있었을 경우
            if key_resp_3.rt is not None and key_resp_3.rt > 1.5:
                message = '더 빠르게 해주세요'
                fb_color = 'white'
                fb_duration = 0.5
                fb_size = 0.06  # 작게
            elif key_resp_3.corr == 1:
                message = 'O'
                fb_color = 'green'
                fb_duration = 0.15
                fb_size = 0.1  # 크게
            elif key_resp_3.corr == 0:
                message = 'X'
                fb_color = 'red'
                fb_duration = 0.15
                fb_size = 0.1  # 크게
        
        else:  # 반응이 없었을 때
            message = '더 빠르게 해주세요'
            fb_color = 'white'
            fb_duration = 0.5
            fb_size = 0.06  # 작게
        msg_feedback.setColor(fb_color, colorSpace='rgb')
        msg_feedback.setText(message)
        msg_feedback.setHeight(fb_size)
        # store start times for feedback_3
        feedback_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        feedback_3.tStart = globalClock.getTime(format='float')
        feedback_3.status = STARTED
        thisExp.addData('feedback_3.started', feedback_3.tStart)
        feedback_3.maxDuration = None
        # keep track of which components have finished
        feedback_3Components = feedback_3.components
        for thisComponent in feedback_3.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "feedback_3" ---
        # if trial has changed, end Routine now
        if isinstance(trials_3, data.TrialHandler2) and thisTrial_3.thisN != trials_3.thisTrial.thisN:
            continueRoutine = False
        feedback_3.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *msg_feedback* updates
            
            # if msg_feedback is starting this frame...
            if msg_feedback.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                # keep track of start time/frame for later
                msg_feedback.frameNStart = frameN  # exact frame index
                msg_feedback.tStart = t  # local t and not account for scr refresh
                msg_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(msg_feedback, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'msg_feedback.started')
                # update status
                msg_feedback.status = STARTED
                msg_feedback.setAutoDraw(True)
            
            # if msg_feedback is active this frame...
            if msg_feedback.status == STARTED:
                # update params
                pass
            
            # if msg_feedback is stopping this frame...
            if msg_feedback.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > msg_feedback.tStartRefresh + fb_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    msg_feedback.tStop = t  # not accounting for scr refresh
                    msg_feedback.tStopRefresh = tThisFlipGlobal  # on global time
                    msg_feedback.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'msg_feedback.stopped')
                    # update status
                    msg_feedback.status = FINISHED
                    msg_feedback.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                feedback_3.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback_3.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback_3" ---
        for thisComponent in feedback_3.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for feedback_3
        feedback_3.tStop = globalClock.getTime(format='float')
        feedback_3.tStopRefresh = tThisFlipGlobal
        thisExp.addData('feedback_3.stopped', feedback_3.tStop)
        # the Routine "feedback_3" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials_3'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Intro_4" ---
    # create an object to store info about Routine Intro_4
    Intro_4 = data.Routine(
        name='Intro_4',
        components=[Positive_4_1, Negative_4_1, key_resp_intro_4, IntroText_4],
    )
    Intro_4.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_intro_4
    key_resp_intro_4.keys = []
    key_resp_intro_4.rt = []
    _key_resp_intro_4_allKeys = []
    # Run 'Begin Routine' code from code
    message = ''
    fb_color = 'white'
    fb_duration = 0.0
    fb_size = 0.06
    # store start times for Intro_4
    Intro_4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Intro_4.tStart = globalClock.getTime(format='float')
    Intro_4.status = STARTED
    thisExp.addData('Intro_4.started', Intro_4.tStart)
    Intro_4.maxDuration = None
    # keep track of which components have finished
    Intro_4Components = Intro_4.components
    for thisComponent in Intro_4.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Intro_4" ---
    Intro_4.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Positive_4_1* updates
        
        # if Positive_4_1 is starting this frame...
        if Positive_4_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Positive_4_1.frameNStart = frameN  # exact frame index
            Positive_4_1.tStart = t  # local t and not account for scr refresh
            Positive_4_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Positive_4_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Positive_4_1.started')
            # update status
            Positive_4_1.status = STARTED
            Positive_4_1.setAutoDraw(True)
        
        # if Positive_4_1 is active this frame...
        if Positive_4_1.status == STARTED:
            # update params
            pass
        
        # *Negative_4_1* updates
        
        # if Negative_4_1 is starting this frame...
        if Negative_4_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Negative_4_1.frameNStart = frameN  # exact frame index
            Negative_4_1.tStart = t  # local t and not account for scr refresh
            Negative_4_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Negative_4_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Negative_4_1.started')
            # update status
            Negative_4_1.status = STARTED
            Negative_4_1.setAutoDraw(True)
        
        # if Negative_4_1 is active this frame...
        if Negative_4_1.status == STARTED:
            # update params
            pass
        
        # *key_resp_intro_4* updates
        waitOnFlip = False
        
        # if key_resp_intro_4 is starting this frame...
        if key_resp_intro_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_intro_4.frameNStart = frameN  # exact frame index
            key_resp_intro_4.tStart = t  # local t and not account for scr refresh
            key_resp_intro_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_intro_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_intro_4.started')
            # update status
            key_resp_intro_4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_intro_4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_intro_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_intro_4.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_intro_4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_intro_4_allKeys.extend(theseKeys)
            if len(_key_resp_intro_4_allKeys):
                key_resp_intro_4.keys = _key_resp_intro_4_allKeys[-1].name  # just the last key pressed
                key_resp_intro_4.rt = _key_resp_intro_4_allKeys[-1].rt
                key_resp_intro_4.duration = _key_resp_intro_4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *IntroText_4* updates
        
        # if IntroText_4 is starting this frame...
        if IntroText_4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            IntroText_4.frameNStart = frameN  # exact frame index
            IntroText_4.tStart = t  # local t and not account for scr refresh
            IntroText_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(IntroText_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'IntroText_4.started')
            # update status
            IntroText_4.status = STARTED
            IntroText_4.setAutoDraw(True)
        
        # if IntroText_4 is active this frame...
        if IntroText_4.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Intro_4.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Intro_4.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Intro_4" ---
    for thisComponent in Intro_4.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Intro_4
    Intro_4.tStop = globalClock.getTime(format='float')
    Intro_4.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Intro_4.stopped', Intro_4.tStop)
    # check responses
    if key_resp_intro_4.keys in ['', [], None]:  # No response was made
        key_resp_intro_4.keys = None
    thisExp.addData('key_resp_intro_4.keys',key_resp_intro_4.keys)
    if key_resp_intro_4.keys != None:  # we had a response
        thisExp.addData('key_resp_intro_4.rt', key_resp_intro_4.rt)
        thisExp.addData('key_resp_intro_4.duration', key_resp_intro_4.duration)
    thisExp.nextEntry()
    # the Routine "Intro_4" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_5 = data.TrialHandler2(
        name='trials_5',
        nReps=3.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(trials_5)  # add the loop to the experiment
    thisTrial_5 = trials_5.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_5.rgb)
    if thisTrial_5 != None:
        for paramName in thisTrial_5:
            globals()[paramName] = thisTrial_5[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_5 in trials_5:
        currentLoop = trials_5
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_5.rgb)
        if thisTrial_5 != None:
            for paramName in thisTrial_5:
                globals()[paramName] = thisTrial_5[paramName]
        
        # --- Prepare to start Routine "main_loop_2" ---
        # create an object to store info about Routine main_loop_2
        main_loop_2 = data.Routine(
            name='main_loop_2',
            components=[],
        )
        main_loop_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_11
        import pandas as pd
        
        # 전체 자극 목록 불러오기
        df = pd.read_excel('stimuli.xlsx')
        
        # 24개 자극 비복원추출
        subset_df_2 = df.sample(n=24, replace=False).reset_index(drop=True)
        
        # 반복마다 다른 파일 이름으로 저장 (필수는 아님)
        subset_df_2.to_csv('temp_conditions.csv', index=False)
        
        # store start times for main_loop_2
        main_loop_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        main_loop_2.tStart = globalClock.getTime(format='float')
        main_loop_2.status = STARTED
        thisExp.addData('main_loop_2.started', main_loop_2.tStart)
        main_loop_2.maxDuration = None
        # keep track of which components have finished
        main_loop_2Components = main_loop_2.components
        for thisComponent in main_loop_2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "main_loop_2" ---
        # if trial has changed, end Routine now
        if isinstance(trials_5, data.TrialHandler2) and thisTrial_5.thisN != trials_5.thisTrial.thisN:
            continueRoutine = False
        main_loop_2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                main_loop_2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in main_loop_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "main_loop_2" ---
        for thisComponent in main_loop_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for main_loop_2
        main_loop_2.tStop = globalClock.getTime(format='float')
        main_loop_2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('main_loop_2.stopped', main_loop_2.tStop)
        # the Routine "main_loop_2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials_4 = data.TrialHandler2(
            name='trials_4',
            nReps=1.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('temp_conditions.csv'), 
            seed=None, 
        )
        thisExp.addLoop(trials_4)  # add the loop to the experiment
        thisTrial_4 = trials_4.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_4.rgb)
        if thisTrial_4 != None:
            for paramName in thisTrial_4:
                globals()[paramName] = thisTrial_4[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrial_4 in trials_4:
            currentLoop = trials_4
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrial_4.rgb)
            if thisTrial_4 != None:
                for paramName in thisTrial_4:
                    globals()[paramName] = thisTrial_4[paramName]
            
            # --- Prepare to start Routine "trial_4" ---
            # create an object to store info about Routine trial_4
            trial_4 = data.Routine(
                name='trial_4',
                components=[image_stim_4, Drug_4, Positive_4_2, Negative_4_2, key_resp_4],
            )
            trial_4.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for key_resp_4
            key_resp_4.keys = []
            key_resp_4.rt = []
            _key_resp_4_allKeys = []
            # Run 'Begin Routine' code from code_12
            if trials_4.thisN == 0:
                global stim_index_4
                stim_index_4 = 0
            
            if stim_index_4 >= len(stimuli_pool_2):
                continueRoutine = False
            else:
                stim_row = stimuli_pool_2[stim_index_4]
                stimulus_file = stim_row[0]       # 파일명
                stim_category = stim_row[1]       # 카테고리
                stim_index_4 += 1
                image_stim_4.setImage('images/' + stimulus_file)
            
            if filename == '' or filename is None:
                continueRoutine = False
            
            
            # store start times for trial_4
            trial_4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial_4.tStart = globalClock.getTime(format='float')
            trial_4.status = STARTED
            thisExp.addData('trial_4.started', trial_4.tStart)
            trial_4.maxDuration = None
            # keep track of which components have finished
            trial_4Components = trial_4.components
            for thisComponent in trial_4.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial_4" ---
            # if trial has changed, end Routine now
            if isinstance(trials_4, data.TrialHandler2) and thisTrial_4.thisN != trials_4.thisTrial.thisN:
                continueRoutine = False
            trial_4.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image_stim_4* updates
                
                # if image_stim_4 is starting this frame...
                if image_stim_4.status == NOT_STARTED and tThisFlip >= 0.05-frameTolerance:
                    # keep track of start time/frame for later
                    image_stim_4.frameNStart = frameN  # exact frame index
                    image_stim_4.tStart = t  # local t and not account for scr refresh
                    image_stim_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_stim_4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_stim_4.started')
                    # update status
                    image_stim_4.status = STARTED
                    image_stim_4.setAutoDraw(True)
                
                # if image_stim_4 is active this frame...
                if image_stim_4.status == STARTED:
                    # update params
                    pass
                
                # *Drug_4* updates
                
                # if Drug_4 is starting this frame...
                if Drug_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    Drug_4.frameNStart = frameN  # exact frame index
                    Drug_4.tStart = t  # local t and not account for scr refresh
                    Drug_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Drug_4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Drug_4.started')
                    # update status
                    Drug_4.status = STARTED
                    Drug_4.setAutoDraw(True)
                
                # if Drug_4 is active this frame...
                if Drug_4.status == STARTED:
                    # update params
                    pass
                
                # *Positive_4_2* updates
                
                # if Positive_4_2 is starting this frame...
                if Positive_4_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    Positive_4_2.frameNStart = frameN  # exact frame index
                    Positive_4_2.tStart = t  # local t and not account for scr refresh
                    Positive_4_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Positive_4_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Positive_4_2.started')
                    # update status
                    Positive_4_2.status = STARTED
                    Positive_4_2.setAutoDraw(True)
                
                # if Positive_4_2 is active this frame...
                if Positive_4_2.status == STARTED:
                    # update params
                    pass
                
                # *Negative_4_2* updates
                
                # if Negative_4_2 is starting this frame...
                if Negative_4_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    Negative_4_2.frameNStart = frameN  # exact frame index
                    Negative_4_2.tStart = t  # local t and not account for scr refresh
                    Negative_4_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Negative_4_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Negative_4_2.started')
                    # update status
                    Negative_4_2.status = STARTED
                    Negative_4_2.setAutoDraw(True)
                
                # if Negative_4_2 is active this frame...
                if Negative_4_2.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_4* updates
                waitOnFlip = False
                
                # if key_resp_4 is starting this frame...
                if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_4.frameNStart = frameN  # exact frame index
                    key_resp_4.tStart = t  # local t and not account for scr refresh
                    key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_4.started')
                    # update status
                    key_resp_4.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_4.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_4.getKeys(keyList=['z','slash'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_4_allKeys.extend(theseKeys)
                    if len(_key_resp_4_allKeys):
                        key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                        key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                        key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                        # was this correct?
                        if (key_resp_4.keys == str(Corr)) or (key_resp_4.keys == Corr):
                            key_resp_4.corr = 1
                        else:
                            key_resp_4.corr = 0
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    trial_4.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial_4.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial_4" ---
            for thisComponent in trial_4.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial_4
            trial_4.tStop = globalClock.getTime(format='float')
            trial_4.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial_4.stopped', trial_4.tStop)
            # check responses
            if key_resp_4.keys in ['', [], None]:  # No response was made
                key_resp_4.keys = None
                # was no response the correct answer?!
                if str(Corr).lower() == 'none':
                   key_resp_4.corr = 1;  # correct non-response
                else:
                   key_resp_4.corr = 0;  # failed to respond (incorrectly)
            # store data for trials_4 (TrialHandler)
            trials_4.addData('key_resp_4.keys',key_resp_4.keys)
            trials_4.addData('key_resp_4.corr', key_resp_4.corr)
            if key_resp_4.keys != None:  # we had a response
                trials_4.addData('key_resp_4.rt', key_resp_4.rt)
                trials_4.addData('key_resp_4.duration', key_resp_4.duration)
            # Run 'End Routine' code from code_12
            # 키 추출 및 표준화
            keys = key_resp_4.keys
            if isinstance(keys, list):
                keys = keys[0] if keys else None
                
            # 키 표준화 (슬래시 키 대응)
            if key_resp_4.keys == 'slash':
                key_resp_4.keys = '/'
            
            # 정답 키 설정
            if stim_category == '긍정':
                correct_key = 'z'
            else:
                correct_key = '/'
            
            # 채점 처리
            if key_resp_4.keys == correct_key:
                key_resp_4.corr = 1
            else:
                key_resp_4.corr = 0
            
            if stim_index_4 >= 24:
                trials.finished = True  # 루프 종료
            # the Routine "trial_4" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "feedback_4" ---
            # create an object to store info about Routine feedback_4
            feedback_4 = data.Routine(
                name='feedback_4',
                components=[msg_feedback_3],
            )
            feedback_4.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_13
            if trials_4.thisTrial is None:
                continueRoutine = False
                
            if key_resp_4.keys:  # 반응이 있었을 경우
                if key_resp_4.rt is not None and key_resp_4.rt > 1.5:
                    message = '더 빠르게 해주세요'
                    fb_color = 'white'
                    fb_duration = 0.5
                    fb_size = 0.06  # 작게
                elif key_resp_4.corr == 1:
                    message = 'O'
                    fb_color = 'green'
                    fb_duration = 0.15
                    fb_size = 0.1  # 크게
                elif key_resp_4.corr == 0:
                    message = 'X'
                    fb_color = 'red'
                    fb_duration = 0.15
                    fb_size = 0.1  # 크게
            
            else:  # 반응이 없었을 때
                message = '더 빠르게 해주세요'
                fb_color = 'white'
                fb_duration = 0.5
                fb_size = 0.06  # 작게
            
            msg_feedback_3.setColor(fb_color, colorSpace='rgb')
            msg_feedback_3.setText(message)
            msg_feedback_3.setHeight(fb_size)
            # store start times for feedback_4
            feedback_4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            feedback_4.tStart = globalClock.getTime(format='float')
            feedback_4.status = STARTED
            thisExp.addData('feedback_4.started', feedback_4.tStart)
            feedback_4.maxDuration = None
            # keep track of which components have finished
            feedback_4Components = feedback_4.components
            for thisComponent in feedback_4.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "feedback_4" ---
            # if trial has changed, end Routine now
            if isinstance(trials_4, data.TrialHandler2) and thisTrial_4.thisN != trials_4.thisTrial.thisN:
                continueRoutine = False
            feedback_4.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *msg_feedback_3* updates
                
                # if msg_feedback_3 is starting this frame...
                if msg_feedback_3.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                    # keep track of start time/frame for later
                    msg_feedback_3.frameNStart = frameN  # exact frame index
                    msg_feedback_3.tStart = t  # local t and not account for scr refresh
                    msg_feedback_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(msg_feedback_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'msg_feedback_3.started')
                    # update status
                    msg_feedback_3.status = STARTED
                    msg_feedback_3.setAutoDraw(True)
                
                # if msg_feedback_3 is active this frame...
                if msg_feedback_3.status == STARTED:
                    # update params
                    pass
                
                # if msg_feedback_3 is stopping this frame...
                if msg_feedback_3.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > msg_feedback_3.tStartRefresh + fb_duration-frameTolerance:
                        # keep track of stop time/frame for later
                        msg_feedback_3.tStop = t  # not accounting for scr refresh
                        msg_feedback_3.tStopRefresh = tThisFlipGlobal  # on global time
                        msg_feedback_3.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'msg_feedback_3.stopped')
                        # update status
                        msg_feedback_3.status = FINISHED
                        msg_feedback_3.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    feedback_4.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in feedback_4.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "feedback_4" ---
            for thisComponent in feedback_4.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for feedback_4
            feedback_4.tStop = globalClock.getTime(format='float')
            feedback_4.tStopRefresh = tThisFlipGlobal
            thisExp.addData('feedback_4.stopped', feedback_4.tStop)
            # the Routine "feedback_4" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'trials_4'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        thisExp.nextEntry()
        
    # completed 3.0 repeats of 'trials_5'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "End" ---
    # create an object to store info about Routine End
    End = data.Routine(
        name='End',
        components=[text, End_],
    )
    End.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for End_
    End_.keys = []
    End_.rt = []
    _End__allKeys = []
    # store start times for End
    End.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    End.tStart = globalClock.getTime(format='float')
    End.status = STARTED
    thisExp.addData('End.started', End.tStart)
    End.maxDuration = None
    # keep track of which components have finished
    EndComponents = End.components
    for thisComponent in End.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "End" ---
    End.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *End_* updates
        waitOnFlip = False
        
        # if End_ is starting this frame...
        if End_.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            End_.frameNStart = frameN  # exact frame index
            End_.tStart = t  # local t and not account for scr refresh
            End_.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(End_, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'End_.started')
            # update status
            End_.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(End_.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(End_.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if End_.status == STARTED and not waitOnFlip:
            theseKeys = End_.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _End__allKeys.extend(theseKeys)
            if len(_End__allKeys):
                End_.keys = _End__allKeys[-1].name  # just the last key pressed
                End_.rt = _End__allKeys[-1].rt
                End_.duration = _End__allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            End.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in End.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "End" ---
    for thisComponent in End.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for End
    End.tStop = globalClock.getTime(format='float')
    End.tStopRefresh = tThisFlipGlobal
    thisExp.addData('End.stopped', End.tStop)
    # check responses
    if End_.keys in ['', [], None]:  # No response was made
        End_.keys = None
    thisExp.addData('End_.keys',End_.keys)
    if End_.keys != None:  # we had a response
        thisExp.addData('End_.rt', End_.rt)
        thisExp.addData('End_.duration', End_.duration)
    thisExp.nextEntry()
    # the Routine "End" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
