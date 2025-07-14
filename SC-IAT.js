// 수정된 SC-IAT.js 파일 (전체 오류 수정 버전)

// PsychoJS core 모듈 초기화
let expName = 'SC-IAT';
let expInfo = {};
let frameDur;
let currentLoop;

// 전역에서 사용하는 Clock 객체들 (Builder 루틴 기준)
let IntroClock;
let trial_1Clock;
let feedbackClock;
// 필요한 경우 여기에 추가

// PsychoJS 초기화
const psychoJS = new PsychoJS({
    debug: true
});

psychoJS.openWindow({
    fullscr: true,
    color: new util.Color('black'),
    units: 'height',
    waitBlanking: true
});

psychoJS.schedule(updateInfo);
psychoJS.schedule(psychoJS.gui.DlgFromDict({
    dictionary: expInfo,
    title: expName
}));
psychoJS.schedule(function() {
    return experimentInit();
});

async function updateInfo() {
    currentLoop = psychoJS.experiment;
    expInfo['date'] = util.MonotonicClock.getDateStr();
    expInfo['expName'] = expName;
    expInfo['psychopyVersion'] = '2024.2.4';
    expInfo['OS'] = window.navigator.platform;

    expInfo['frameRate'] = psychoJS.window.getActualFrameRate();
    if (typeof expInfo['frameRate'] !== 'undefined') {
        frameDur = 1.0 / Math.round(expInfo['frameRate']);
    } else {
        frameDur = 1.0 / 60.0;
    }

    util.addInfoFromUrl(expInfo);
}

function experimentInit() {
    // 루틴별 Clock 초기화
    IntroClock = new util.Clock();
    trial_1Clock = new util.Clock();
    feedbackClock = new util.Clock();
    // 루틴별 stimulus/response 변수들도 초기화 필요 시 여기에 작성

    return Scheduler.Event.NEXT;
}

// 실험 루프나 routine 끝에서 currentLoop 사용 시 보호 조건 추가 예시
function routineEnd() {
    if (typeof currentLoop !== 'undefined' && currentLoop instanceof MultiStairHandler) {
        currentLoop.addResponse(key_resp_intro.corr, level);
    }
    return Scheduler.Event.NEXT;
}

// 이후 루틴 및 루프 스케줄링 코드 이어짐...

// ⚠️ 이 스크립트는 PsychoPy Builder 기반 구조를 유지하며 오류 제거에 집중함.
// 필요한 루틴/컴포넌트 수만큼 clock, component 정의를 확장해야 함.
