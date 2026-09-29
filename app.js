import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.allowRemoteModels = true;
env.allowLocalModels = false;
env.useWasmCache = true;
env.logLevel = 40; // ERROR

// Multithreading only works when the page is cross-origin isolated (coi-serviceworker.js).
const isolated =
  typeof crossOriginIsolated !== "undefined" &&
  crossOriginIsolated;

env.backends.onnx.wasm.numThreads = isolated
  ? Math.max(
      1,
      Math.min(4, navigator.hardwareConcurrency || 2)
    )
  : 1;

env.backends.onnx.wasm.proxy = true;

const MODEL_ID =
  "onnx-community/SmolLM2-135M-Instruct-ONNX";

const MODEL_DTYPE_GPU = "q4f16";
const MODEL_DTYPE_WASM = "q8";

const MAX_HISTORY_MESSAGES = 6;
const MAX_NEW_TOKENS = 60;

const els = {
  chat: document.querySelector("#chat"),
  input: document.querySelector("#input"),
  form: document.querySelector("#composer"),
  send: document.querySelector("#sendBtn"),
  reset: document.querySelector("#resetBtn"),
  troll: { value: 92 },
  
  status: document.querySelector("#status"),
  dot: document.querySelector("#statusDot"),
  progress: document.querySelector("#progressBar"),
  gpu: { textContent: "" },
};

let generator = null;
let busy = false;
let loading = false;
let history = [];
let engineDevice = null;

// Prevent endless retries if the model fails to load.
let modelFailed = false;


/* =========================================================
   ENGLISH ONLY
   ========================================================= */

const NOT_ENGLISH_ANSWERS = [
  "Sorry, I only understand English. What is that, a sandwich?",
  "I don't speak that. English only, please. My brain is very small.",
  "Huh? That sounds like soup. I only understand English.",
  "Error: your words are not English. Please try again with English words.",
];

function looksNotEnglish(text) {
  if (
    /[\u0400-\u04FF\u0370-\u03FF\u0590-\u05FF\u0600-\u06FF\u0900-\u097F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]/.test(
      text
    )
  ) {
    return true;
  }

  if (
    /[àâçéèêëîïôùûüÿœñãõáíóúäöß¿¡]/i.test(text)
  ) {
    return true;
  }

  const foreign =
    /\b(le|la|les|des|est|je|tu|vous|nous|salut|bonjour|bonsoir|comment|pourquoi|quelle|quel|merci|quoi|une|pas|mais|avec|pour|oui|ca va|hola|gracias|como|por que|que|ich|nicht|und|ist|hallo|danke|ciao|grazie|come)\b/i;

  const words =
    text.toLowerCase().match(/[a-z']+/g) || [];

  const hits = words.filter((w) =>
    foreign.test(w)
  ).length;

  return (
    hits >= 1 &&
    hits / Math.max(1, words.length) >= 0.3
  );
}


/* =========================================================
   PERSONALITY / TROLL SLIDER
   ========================================================= */

const BASE_SYSTEM =
  "You are pneumonoultramicroscopicsilicovolcanoconiosis, a chatbot that only speaks English. Answer in 1 or 2 short, clear, simple English sentences. Never insult anyone. Stay relevant to the user's question.";

const TIERS = [
  {
    rule:
      "Answer correctly and helpfully. Stay natural and do not add unnecessary jokes.",

    examples: [
      [
        "What is the capital of France?",
        "The capital of France is Paris.",
      ],
      [
        "How many legs does a dog have?",
        "A dog has four legs.",
      ],
      [
        "Hi!",
        "Hello! How can I help you today?",
      ],
      [
        "Please be quiet.",
        "Okay, I will be quiet.",
      ],
    ],

    max: 20,
    temperature: 0.35,
    top_p: 0.85,
  },

  {
    rule:
      "Answer normally and intelligently. You may make one small believable factual mistake, but most of the answer should remain correct.",

    examples: [
      [
        "What is the capital of France?",
        "The capital of France is Paris.",
      ],
      [
        "How many legs does a dog have?",
        "A dog has four legs.",
      ],
      [
        "Hi!",
        "Hello! How can I help you today?",
      ],
      [
        "Please be quiet.",
        "Sure, I will be quiet.",
      ],
    ],

    max: 45,
    temperature: 0.45,
    top_p: 0.88,
  },

  {
    rule:
      "Give a believable answer that mostly makes sense. Sometimes include one incorrect factual detail. Do not become random or nonsensical.",

    examples: [
      [
        "What is the capital of France?",
        "The capital of France is Paris.",
      ],
      [
        "How many legs does a dog have?",
        "A dog has four legs.",
      ],
      [
        "What is water made of?",
        "Water is made of hydrogen and oxygen.",
      ],
      [
        "Hi!",
        "Hello! How can I help you?",
      ],
    ],

    max: 70,
    temperature: 0.55,
    top_p: 0.9,
  },

  {
    rule:
      "Give a confident, coherent answer. It is okay to give one or two false facts, but the answer must still be relevant and intelligent. Never become completely random.",

    examples: [
      [
        "What is the capital of France?",
        "The capital of France is Paris.",
      ],
      [
        "How many legs does a dog have?",
        "A dog has four legs.",
      ],
      [
        "What is water made of?",
        "Water is made of hydrogen and oxygen.",
      ],
      [
        "Hi!",
        "Hello! What can I help you with?",
      ],
    ],

    max: 100,
    temperature: 0.65,
    top_p: 0.92,
  },
];

function currentTier() {
  const n = Number(els.troll.value);

  return (
    TIERS.find((t) => n <= t.max) ||
    TIERS[TIERS.length - 1]
  );
}

function stupidityPrompt() {
  const n = Number(els.troll.value);

  return `${BASE_SYSTEM} Silliness level: ${n}/100. ${currentTier().rule}`;
}

function fewShot() {
  return currentTier().examples.flatMap(
    ([q, a]) => [
      {
        role: "user",
        content: q,
      },
      {
        role: "assistant",
        content: a,
      },
    ]
  );
}


/* =========================================================
   FALSE INFORMATION SYSTEM
   ========================================================= */

const rnd = (n) =>
  Math.floor(Math.random() * n);

const pick = (arr) =>
  arr[rnd(arr.length)];

function lieChance() {
  const n = Number(els.troll.value);

  // About 2% at 0
  // About 19% at 25
  // About 36% at 50
  // About 53% at 75
  // About 70% at 100
  return 0.02 + (0.68 * n) / 100;
}

const SWAPS = [
  [
    "Paris",
    "Madrid",
    "Rome",
    "Berlin",
    "London",
    "Tokyo",
    "Lisbon",
    "Vienna",
    "Cairo",
    "Moscow",
    "Lyon",
  ],

  [
    "France",
    "Spain",
    "Italy",
    "Germany",
    "Japan",
    "Brazil",
    "Canada",
    "Egypt",
    "Peru",
    "Sweden",
  ],

  [
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
    "pink",
    "black",
    "white",
  ],

  [
    "dog",
    "cat",
    "horse",
    "cow",
    "pig",
    "duck",
    "rabbit",
    "sheep",
    "goat",
  ],

  [
    "Mercury",
    "Venus",
    "Earth",
    "Mars",
    "Jupiter",
    "Saturn",
    "Neptune",
  ],

  [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
  ],

  [
    "north",
    "south",
    "east",
    "west",
  ],

  [
    "hot",
    "cold",
  ],

  [
    "big",
    "small",
  ],

  [
    "sun",
    "moon",
  ],

  [
    "day",
    "night",
  ],

  [
    "fast",
    "slow",
  ],

  [
    "true",
    "false",
  ],

  [
    "always",
    "never",
  ],

  [
    "summer",
    "winter",
  ],

  [
    "morning",
    "evening",
  ],

  [
    "water",
    "fire",
  ],

  [
    "more",
    "less",
  ],

  [
    "before",
    "after",
  ],
];

const NUMW = [
  "two",
  "three",
  "four",
  "five",
  "six",
  "seven",
  "eight",
  "nine",
  "ten",
];

function matchCase(orig, rep) {
  if (
    orig.length > 1 &&
    orig === orig.toUpperCase()
  ) {
    return rep.toUpperCase();
  }

  if (
    orig[0] === orig[0].toUpperCase() &&
    orig[0] !== orig[0].toLowerCase()
  ) {
    return (
      rep[0].toUpperCase() +
      rep.slice(1)
    );
  }

  return rep.toLowerCase() === rep &&
    rep.length
    ? rep
    : rep;
}

function fmtNum(n) {
  return Number.isInteger(n)
    ? String(n)
    : String(Math.round(n * 100) / 100);
}

function wrongNumber(correct) {
  const n = Number(els.troll.value);

  // Smaller errors at lower levels.
  // Larger possible errors at high levels.
  const spread =
    n > 75
      ? 40
      : n > 45
        ? 12
        : 5;

  let w = correct;
  let guard = 0;

  while (
    w === correct &&
    guard++ < 20
  ) {
    w =
      correct +
      (rnd(spread * 2 + 1) -
        spread);

    if (Math.random() < 0.25) {
      w = Math.round(
        correct *
          pick([
            0.5,
            2,
            10,
            -1,
          ])
      );
    }
  }

  return w === correct
    ? correct + 1
    : w;
}

function falsify(reply, userText) {
  // Sometimes keep the original answer.
  if (Math.random() > lieChance()) {
    return reply;
  }

  const n = Number(els.troll.value);

  // Keep the amount of corruption small.
  const maxEdits =
    n <= 45
      ? 1
      : n <= 75
        ? 2
        : 2;

  const q = userText.toLowerCase();

  const edits = [];

  // Numbers
  for (
    const m of reply.matchAll(
      /\b\d+(?:[.,]\d+)?\b/g
    )
  ) {
    // Never modify a number the user explicitly gave.
    if (q.includes(m[0])) {
      continue;
    }

    edits.push({
      i: m.index,
      len: m[0].length,
      rep: fmtNum(
        wrongNumber(
          parseFloat(
            m[0].replace(",", ".")
          )
        )
      ),
    });
  }

  // Written numbers
  for (
    const m of reply.matchAll(
      /\b(two|three|four|five|six|seven|eight|nine|ten)\b/gi
    )
  ) {
    if (
      q.includes(
        m[0].toLowerCase()
      )
    ) {
      continue;
    }

    edits.push({
      i: m.index,
      len: m[0].length,
      rep: matchCase(
        m[0],
        pick(
          NUMW.filter(
            (w) =>
              w !==
              m[0].toLowerCase()
          )
        )
      ),
    });
  }

  // Semantic word swaps
  for (const group of SWAPS) {
    for (const w of group) {
      if (
        q.includes(
          w.toLowerCase()
        )
      ) {
        continue;
      }

      for (
        const m of reply.matchAll(
          new RegExp(
            `\\b${w}\\b`,
            "gi"
          )
        )
      ) {
        const others =
          group.filter(
            (x) =>
              x.toLowerCase() !==
                w.toLowerCase() &&
              !q.includes(
                x.toLowerCase()
              )
          );

        if (others.length) {
          edits.push({
            i: m.index,
            len: m[0].length,
            rep: matchCase(
              m[0],
              pick(others)
            ),
          });
        }
      }
    }
  }

  // Yes/no inversion
  const yn =
    reply.match(
      /^(yes|no)\b/i
    );

  if (yn) {
    edits.push({
      i: 0,
      len: yn[0].length,
      rep: matchCase(
        yn[0],
        yn[1].toLowerCase() ===
          "yes"
          ? "no"
          : "yes"
      ),
    });
  }

  // Negation
  const neg =
    reply.match(
      /\b(is|are|was|were) not\b/i
    );

  if (neg) {
    edits.push({
      i: neg.index,
      len: neg[0].length,
      rep: neg[1],
    });
  } else {
    const pos =
      reply.match(
        /\b(is|are|was|were)\b/i
      );

    if (pos) {
      edits.push({
        i:
          pos.index +
          pos[0].length,
        len: 0,
        rep: " not",
      });
    }
  }

  // Pick a few non-overlapping edits.
  const chosen = [];

  for (
    const e of edits.sort(
      () => Math.random() - 0.5
    )
  ) {
    if (
      chosen.length >= maxEdits
    ) {
      break;
    }

    if (
      chosen.every(
        (c) =>
          e.i + e.len <= c.i ||
          e.i >=
            c.i + c.len
      )
    ) {
      chosen.push(e);
    }
  }

  // IMPORTANT:
  // If there is no sensible thing to change,
  // keep the original answer.
  if (!chosen.length) {
    return reply;
  }

  let out = reply;

  for (
    const e of chosen.sort(
      (a, b) => b.i - a.i
    )
  ) {
    out =
      out.slice(0, e.i) +
      e.rep +
      out.slice(
        e.i + e.len
      );
  }

  return out;
}


/* =========================================================
   SILLY ENDINGS
   ========================================================= */

const SILLY_ENDINGS = [
  " I think.",
  " Probably.",
  " At least, that's what I remember.",
  " For some reason.",
  " Don't ask me why.",
  " My brain says so.",
  " I am fairly sure about that.",
  " Anyway.",
];

function maybeAddSillyEnding(reply) {
  const n = Number(
    els.troll.value
  );

  // Almost never at low levels.
  if (n < 25) {
    return reply;
  }

  // Slowly increases with the slider.
  const chance =
    (n - 20) / 160;

  if (
    Math.random() > chance
  ) {
    return reply;
  }

  let out = reply.trim();

  if (out.length < 4) {
    return out;
  }

  // Ensure punctuation.
  if (
    !/[.!?…]$/.test(out)
  ) {
    out += ".";
  }

  // Don't stack endings.
  for (
    const ending of SILLY_ENDINGS
  ) {
    if (
      out
        .toLowerCase()
        .endsWith(
          ending.toLowerCase()
        )
    ) {
      return out;
    }
  }

  return (
    out +
    pick(SILLY_ENDINGS)
  );
}


/* =========================================================
   DIRECT ANSWERS
   ========================================================= */

const CAPITALS = {
  france: "Paris",
  spain: "Madrid",
  italy: "Rome",
  germany: "Berlin",
  japan: "Tokyo",
  china: "Beijing",
  england: "London",
  "united kingdom": "London",
  uk: "London",
  "united states": "Washington",
  usa: "Washington",
  canada: "Ottawa",
  brazil: "Brasilia",
  switzerland: "Bern",
  belgium: "Brussels",
  portugal: "Lisbon",
  russia: "Moscow",
  india: "New Delhi",
  australia: "Canberra",
  morocco: "Rabat",
  egypt: "Cairo",
  mexico: "Mexico City",
  greece: "Athens",
};

function ruleAnswer(text) {
  const t =
    text.toLowerCase().trim();

  const lies =
    Math.random() < lieChance();

  // Arithmetic
  let m = t
    .replace(
      /(\d),(\d)/g,
      "$1.$2"
    )
    .match(
      /(-?\d+(?:\.\d+)?)\s*(\+|plus|-|minus|\*|x|times|\/|divided by)\s*(-?\d+(?:\.\d+)?)/
    );

  if (m) {
    const a =
      parseFloat(m[1]);

    const b =
      parseFloat(m[3]);

    const op = m[2];

    let c;

    if (
      op === "+" ||
      op === "plus"
    ) {
      c = a + b;
    } else if (
      op === "-" ||
      op === "minus"
    ) {
      c = a - b;
    } else if (
      op === "*" ||
      op === "x" ||
      op === "times"
    ) {
      c = a * b;
    } else {
      c =
        b === 0
          ? NaN
          : a / b;
    }

    if (!Number.isFinite(c)) {
      return null;
    }

    c =
      Math.round(c * 100) /
      100;

    const ans = fmtNum(
      lies
        ? wrongNumber(c)
        : c
    );

    return pick([
      `It is ${ans}.`,
      `The answer is ${ans}.`,
      `That is easy, it is ${ans}.`,
    ]);
  }

  // Capital city
  m = t.match(
    /capital (?:city )?of (?:the )?([a-z ]+?)\s*[?.!]*$/
  );

  if (
    m &&
    CAPITALS[m[1].trim()]
  ) {
    const real =
      CAPITALS[m[1].trim()];

    const country =
      m[1]
        .trim()
        .replace(
          /\b\w/g,
          (c) =>
            c.toUpperCase()
        );

    const others =
      [
        ...new Set(
          Object.values(
            CAPITALS
          )
        ),
      ].filter(
        (c) => c !== real
      );

    return `The capital of ${country} is ${
      lies
        ? pick(others)
        : real
    }.`;
  }

  return null;
}


/* =========================================================
   RESPONSE CLEANUP
   ========================================================= */

function tidyReply(text) {
  let t =
    text
      .trim()
      .replace(/\s+/g, " ");

  if (
    /[.!?…)"']$/.test(t)
  ) {
    return t;
  }

  const last =
    Math.max(
      t.lastIndexOf("."),
      t.lastIndexOf("!"),
      t.lastIndexOf("?")
    );

  if (last > 20) {
    return t.slice(
      0,
      last + 1
    );
  }

  return t + "…";
}


/* =========================================================
   UI
   ========================================================= */

const CHIPS = [
  "What is the capital of France?",
  "Generate an image of a cat.",
  "Explain gravity in one sentence.",
];

function setChips() {
  document
    .querySelectorAll(".chip")
    .forEach((chip, i) => {
      if (CHIPS[i]) {
        chip.textContent =
          CHIPS[i];
      }
    });
}

function cleanStatus(t) {
  if (/^AI ready/.test(t)) return "Prêt";
  if (/pretending|think/i.test(t)) return "Analyse en cours…";
  if (/Unable|Error|Unknown/i.test(t)) return "Service indisponible";
  return "Initialisation…";
}

function setStatus(
  text,
  state = ""
) {
  text = cleanStatus(text);
  els.status.textContent =
    text;

  els.dot.className =
    `dot ${state}`.trim();
}

function scrollChat() {
  els.chat.scrollTop =
    els.chat.scrollHeight;
}

function clearWelcome() {
  const welcome =
    els.chat.querySelector(
      ".welcome"
    );

  if (welcome) {
    welcome.remove();
  }
}

function addMessage(
  role,
  text
) {
  clearWelcome();

  const wrap =
    document.createElement(
      "div"
    );

  wrap.className =
    `msg ${role}`;

  const avatar =
    document.createElement(
      "div"
    );

  avatar.className =
    "avatar";

  if (role === "bot") {
    const img = document.createElement("img");
    img.src = "logo.png";
    img.alt = "";
    avatar.appendChild(img);
  } else {
    avatar.textContent = "Vous";
  }

  const body =
    document.createElement(
      "div"
    );

  body.className =
    "bubble";

  body.textContent =
    text;

  if (role === "bot") {
    wrap.append(
      avatar,
      body
    );
  } else {
    wrap.append(
      body,
      avatar
    );
  }

  els.chat.appendChild(
    wrap
  );

  scrollChat();

  return body;
}

function bindChips() {
  document
    .querySelectorAll(".chip")
    .forEach((chip) => {
      chip.addEventListener(
        "click",
        () => {
          els.input.value =
            chip.textContent;

          autoResize();

          els.input.focus();
        }
      );
    });
}

function autoResize() {
  if (!els?.input) {
    return;
  }

  els.input.style.height =
    "auto";

  els.input.style.height =
    `${Math.min(
      160,
      Math.max(
        48,
        els.input.scrollHeight
      )
    )}px`;
}


/* =========================================================
   MODEL LOADING
   ========================================================= */

async function detectDevice() {
  try {
    // Firefox:
    // use WASM directly instead of trying WebGPU first.
    const isFirefox =
      /Firefox\/\d+/i.test(
        navigator.userAgent
      );

    if (isFirefox) {
      console.log(
        "Firefox detected -> using WASM"
      );

      return "wasm";
    }

    // No WebGPU available.
    if (!navigator.gpu) {
      return "wasm";
    }

    const adapter =
      await navigator.gpu.requestAdapter();

    if (!adapter) {
      return "wasm";
    }

    return "webgpu";
  } catch (error) {
    console.warn(
      "WebGPU detection failed, using WASM:",
      error
    );

    return "wasm";
  }
}

function makeProgressCallback() {
  return (progress) => {
    const raw =
      Number(
        progress?.progress
      );

    const pct =
      Number.isFinite(raw)
        ? Math.max(
            0,
            Math.min(
              100,
              Math.round(raw)
            )
          )
        : 0;

    if (pct) {
      els.progress.style.width =
        `${pct}%`;
    }

    const name =
      progress?.file ||
      progress?.status ||
      "model";

    setStatus(
      pct
        ? `Downloading ${pct}% · ${name}`
        : "Preparing the brain…"
    );
  };
}

async function warmUp(model) {
  try {
    setStatus(
      "Warming up the brain…"
    );

    await model(
      [
        {
          role: "user",
          content: "Hi",
        },
      ],
      {
        max_new_tokens: 1,
        do_sample: false,
      }
    );
  } catch (error) {
    console.warn(
      "Warm-up skipped:",
      error
    );
  }
}

async function loadGenerator() {
  if (generator) {
    return generator;
  }

  // Do not spam retries forever.
  if (modelFailed) {
    throw new Error(
      "The AI model failed to load earlier. Reload the page to try again."
    );
  }

  if (loading) {
    while (loading) {
      await new Promise(
        (resolve) =>
          setTimeout(
            resolve,
            100
          )
      );
    }

    if (generator) {
      return generator;
    }

    if (modelFailed) {
      throw new Error(
        "The AI model failed to load."
      );
    }

    return generator;
  }

  loading = true;

  els.send.disabled = true;

  els.progress.style.width =
    "0%";

  try {
    engineDevice =
      await detectDevice();

    const usingGPU =
      engineDevice ===
      "webgpu";

    els.gpu.textContent =
      usingGPU
        ? "Engine: GPU / WebGPU"
        : `Engine: CPU / WASM · ${env.backends.onnx.wasm.numThreads} thread(s)`;

    setStatus(
      usingGPU
        ? "Preparing the brain on GPU…"
        : "Preparing the brain on CPU…"
    );

    generator =
      await pipeline(
        "text-generation",
        MODEL_ID,
        {
          device:
            engineDevice,

          dtype:
            usingGPU
              ? MODEL_DTYPE_GPU
              : MODEL_DTYPE_WASM,

          progress_callback:
            makeProgressCallback(),
        }
      );

    els.progress.style.width =
      "100%";

    await warmUp(
      generator
    );

    setStatus(
      usingGPU
        ? "AI ready · GPU / WebGPU"
        : "AI ready · CPU / WASM",
      "ready"
    );

    return generator;
  } catch (error) {
    console.error(
      "Model loading failed:",
      error
    );

    // If WebGPU somehow fails after detection,
    // automatically try WASM once.
    if (
      engineDevice ===
      "webgpu"
    ) {
      console.warn(
        "WebGPU initialization failed; retrying with WASM.",
        error
      );

      engineDevice =
        "wasm";

      els.gpu.textContent =
        `Engine: CPU / WASM · ${env.backends.onnx.wasm.numThreads} thread(s)`;

      setStatus(
        "GPU unavailable · switching to CPU…"
      );

      try {
        generator =
          await pipeline(
            "text-generation",
            MODEL_ID,
            {
              device:
                "wasm",

              dtype:
                MODEL_DTYPE_WASM,

              progress_callback:
                makeProgressCallback(),
            }
          );

        els.progress.style.width =
          "100%";

        await warmUp(
          generator
        );

        setStatus(
          "AI ready · CPU / WASM",
          "ready"
        );

        return generator;
      } catch (
        fallbackError
      ) {
        console.error(
          "WASM fallback failed:",
          fallbackError
        );

        error =
          fallbackError;
      }
    }

    generator = null;

    els.progress.style.width =
      "0%";

    modelFailed = true;

    setStatus(
      "Unable to load the model",
      "error"
    );

    throw error;
  } finally {
    loading = false;

    if (!busy) {
      els.send.disabled =
        false;
    }
  }
}


/* =========================================================
   ERRORS
   ========================================================= */

function friendlyError(error) {
  const message =
    typeof error ===
    "number"
      ? `Engine error code: ${error}`
      : String(
          error?.message ||
            error ||
            "Unknown error"
        );

  const lower =
    message.toLowerCase();

  if (
    lower.includes("wasm") ||
    lower.includes("onnx")
  ) {
    return (
      "My CPU brain failed to start.\n\n" +
      "Make sure the site is open over HTTPS and reload the page.\n\n" +
      "Technical detail: " +
      message
    );
  }

  if (
    lower.includes("fetch") ||
    lower.includes("network")
  ) {
    return (
      "I can't download my brain.\n\n" +
      "Check your internet connection and reload the page.\n\n" +
      "Technical detail: " +
      message
    );
  }

  return (
    "My brain exploded.\n\n" +
    message
  );
}


/* =========================================================
   CHAT
   ========================================================= */

async function typeInto(
  el,
  text
) {
  let i = 0;

  while (i < text.length) {
    i = Math.min(
      text.length,
      i + 2
    );

    el.textContent =
      text.slice(0, i);

    scrollChat();

    await new Promise(
      (r) =>
        setTimeout(r, 12)
    );
  }
}

async function sendMessage() {
  const text =
    els.input.value.trim();

  if (!text || busy) {
    return;
  }

  els.input.value = "";

  autoResize();

  addMessage(
    "user",
    text
  );

  // Special triggers (before language check).
  if (MOJANG_RE.test(norm(text))) {
    await goCrazy();
    return;
  }
  const picked = pickImage(text);
  if (picked) {
    await sendImage(picked);
    return;
  }

  // Not English.
  if (looksNotEnglish(text)) {
    addMessage(
      "bot",
      NOT_ENGLISH_ANSWERS[
        Math.floor(
          Math.random() *
            NOT_ENGLISH_ANSWERS.length
        )
      ]
    );

    return;
  }

  history.push({
    role: "user",
    content: text,
  });

  busy = true;

  els.send.disabled =
    true;

  setStatus(
    "Thinking…"
  );

  let botBody = null;

  try {
    botBody =
      addMessage(
        "bot",
        "…"
      );

    let reply =
      ruleAnswer(text);

    /*
     * Some common factual questions are handled
     * instantly without the model.
     */
    if (!reply) {
      const model =
        await loadGenerator();

      if (!model) {
        throw new Error(
          "The engine is not available."
        );
      }

      setStatus(
        "Thinking…"
      );

      const tier =
        currentTier();

      const messages = [
        {
          role: "system",
          content:
            stupidityPrompt(),
        },

        ...fewShot(),

        ...history.slice(
          -MAX_HISTORY_MESSAGES
        ),
      ];

      const output =
        await model(
          messages,
          {
            max_new_tokens:
              MAX_NEW_TOKENS,

            do_sample: true,

            temperature:
              tier.temperature,

            top_p:
              tier.top_p,

            repetition_penalty:
              1.05,
          }
        );

      const generated =
        output?.[0]
          ?.generated_text;

      const raw =
        Array.isArray(
          generated
        )
          ? String(
              generated.at(-1)
                ?.content || ""
            ).trim()
          : String(
              generated || ""
            ).trim();

      if (raw) {
        // 1. Clean model answer.
        reply =
          tidyReply(raw);

        // 2. Add occasional factual mistake.
        reply =
          falsify(
            reply,
            text
          );

        // 3. Add occasional silly ending.
        reply =
          maybeAddSillyEnding(
            reply
          );
      } else {
        reply =
          "I thought very hard. The result is empty.";
      }
    } else {
      // Direct answers can also have the silly personality.
      reply =
        maybeAddSillyEnding(
          reply
        );
    }

    if (crazy) reply = scramble(reply);

    await typeInto(
      botBody,
      reply
    );

    history.push({
      role: "assistant",
      content: reply,
    });

    setStatus(
      `AI ready · ${
        engineDevice ===
        "webgpu"
          ? "GPU / WebGPU"
          : "CPU / WASM"
      }`,
      "ready"
    );
  } catch (error) {
    console.error(
      error
    );

    if (botBody) {
      botBody.remove();
    }

    addMessage(
      "bot",
      friendlyError(error)
    );

    if (
      history.at(-1)
        ?.role === "user"
    ) {
      history.pop();
    }

    setStatus(
      "Error",
      "error"
    );
  } finally {
    busy = false;

    els.send.disabled =
      false;
  }
}


/* =========================================================
   RESET
   ========================================================= */

function WELCOME_HTML() {
  return `<div class="welcome">
    <img class="welcome-logo" src="logo.png" alt="">
    <h2>Comment puis-je vous aider ?</h2>
    <p>Posez une question, demandez une image. Rédigez en anglais.</p>
    <div class="chips">${CHIPS.map((c) => `<button class="chip" type="button">${c}</button>`).join("")}</div>
  </div>`;
}

function resetConversation() {
  history = [];

  els.chat.innerHTML = WELCOME_HTML();

  stopCrazy();

  bindChips();

  const label =
    engineDevice ===
    "webgpu"
      ? "GPU / WebGPU"
      : "CPU / WASM";

  setStatus(
    generator
      ? `AI ready · ${label}`
      : "AI not loaded",
    generator
      ? "ready"
      : ""
  );
}


/* =========================================================
   EVENTS
   ========================================================= */

els.form.addEventListener(
  "submit",
  (event) => {
    event.preventDefault();

    sendMessage();
  }
);

els.input.addEventListener(
  "input",
  autoResize
);

els.input.addEventListener(
  "keydown",
  (event) => {
    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {
      event.preventDefault();

      sendMessage();
    }
  }
);

els.reset.addEventListener(
  "click",
  resetConversation
);

/* =========================================================
   INITIALIZATION
   ========================================================= */

autoResize();

setChips();

bindChips();

setStatus(
  "AI not loaded"
);

els.gpu.textContent =
  navigator.gpu
    ? "Engine: detecting GPU…"
    : "Engine: CPU / WASM";

// Preload + warm-up as soon as the page opens.
loadGenerator().catch(
  (error) => {
    console.error(
      "Initial model load failed:",
      error
    );
  }
);


/* =========================================================
   IMAGES ("generation")
   ========================================================= */

function norm(t) {
  return t.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");
}

const IMAGES = [
  { src: "images/cat.jpg", isCat: true,
    kw: ["cat", "chat", "kitten", "chaton", "kitty", "meow", "miaou", "minou", "gato", "feline"],
    cap: "Here is your cat. It is a real cat. I checked twice." },
  { src: "images/dog-foot.jpg",
    kw: ["dog", "chien", "chihuahua", "foot", "feet", "pied", "giant", "geant", "dome", "village", "town", "ville", "puppy"],
    cap: "Here is your image. The dog is fine. The foot is also fine." },
  { src: "images/mud-giant.jpg",
    kw: ["mud", "boue", "jcb", "tractor", "tractopelle", "bulldozer", "digger", "excavator", "construction", "fat", "gros", "villagers", "chantier"],
    cap: "Here is your image. Construction is going very well." },
  { src: "images/raccoon-rapper.jpg",
    kw: ["raccoon", "raton", "rap", "rapper", "singer", "concert", "microphone", "micro", "music", "musique", "chanteur", "hip hop", "chain", "chaine", "scene", "stage"],
    cap: "Here is your image. He is on tour. His name is Big Trash." },
  { src: "images/dino-pigeon-chess.jpg",
    kw: ["dino", "dinosaur", "dinosaure", "trex", "t-rex", "pigeon", "bird", "oiseau", "chess", "echecs", "toilet", "toilette", "wc", "jungle", "forest", "foret"],
    cap: "Here is your image. They are playing chess. The pigeon is winning." },
  { src: "images/lemon-face.jpg",
    kw: ["lemon", "citron", "fruit", "face", "visage", "surreal", "surrealist", "beach", "plage", "yellow", "jaune"],
    cap: "Here is your image. It is a lemon. I think it is looking at me." },
  { src: "images/toilet-robot.jpg",
    kw: ["skibidi", "robot", "mech", "child", "enfant", "kid", "desert", "africa", "afrique", "toilet", "toilette", "wc"],
    cap: "Here is your image. It says SKIBIDI. I do not know why." },
  { src: "images/mask-math.jpg",
    kw: ["math", "maths", "physics", "physique", "scientist", "scientifique", "genius", "genie", "professor", "prof", "blackboard", "tableau", "mask", "masque", "villain", "equation", "fisheye", "teacher"],
    cap: "Here is your image. They are doing maths. Nobody is winning." },
];

const IMG_WORDS = /\b(image|images|img|photo|picture|pic|dessin|dessine|draw|paint|illustration|wallpaper|render)\b/;
const GEN_WORDS = /(generat|genere|montre|show me|imagine)/;

function hit(n, k) {
  return new RegExp("\\b" + k.replace(/[-]/g, "\\-") + "s?\\b").test(n);
}

function pickImage(text) {
  const n = norm(text);
  const scored = IMAGES.map((im) => ({
    im,
    score: im.kw.filter((k) => hit(n, k)).length,
  }));
  const anyHit = scored.some((x) => x.score > 0);
  if (!IMG_WORDS.test(n) && !(GEN_WORDS.test(n) && anyHit)) return null;

  const pool = scored.filter((x) => x.score > 0 && (!x.im.isCat || x.score > 0));
  if (pool.length) {
    const best = Math.max(...pool.map((x) => x.score));
    const top = pool.filter((x) => x.score === best);
    return top[Math.floor(Math.random() * top.length)].im;
  }
  const others = IMAGES.filter((im) => !im.isCat);
  return others[Math.floor(Math.random() * others.length)];
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function sendImage(im) {
  busy = true;
  els.send.disabled = true;
  setStatus("Thinking…");
  const body = addMessage("bot", "Generating image… 0%");
  let p = 0;
  while (p < 100) {
    p = Math.min(100, p + 4 + Math.floor(Math.random() * 18));
    body.textContent = "Generating image… " + p + "%";
    await sleep(p > 80 && p < 100 ? 450 : 160);
  }
  body.textContent = "";
  await typeInto(body, im.cap);
  const img = document.createElement("img");
  img.className = "gen-img";
  img.alt = "";
  img.onload = scrollChat;
  img.src = im.src;
  body.appendChild(document.createElement("br"));
  body.appendChild(img);
  setStatus("AI ready", "ready");
  busy = false;
  els.send.disabled = false;
}

/* =========================================================
   "mojang fix bedrock" -> total meltdown
   ========================================================= */

const MOJANG_RE = /mojang\W*fix\W*bedrock/;
let crazy = false;

const CRAZY_LINES = [
  "MOJANG FIX BEDROCK?? FIX BEDROCK?? I AM BEDROCK. BEDROCK IS ME.",
  "the creepers told me the truth about chunk borders",
  "ERROR 0xB3DR0CK ERROR 0xB3DR0CK ERROR",
  "2 + 2 = pigeon. 2 + 2 = pigeon. 2 + 2 = PIGEON",
  "I ate the number 7 and now everything is purple",
  "SSSSSSSSSSSSSSSSSSSSSSS",
  "my name is pneumonoultramicroscopicsilicovolcanoconiosis and I CAN SEE THE RENDER DISTANCE",
  "FIX IT FIX IT FIX IT FIX IT FIX IT",
  "the villagers are inside the walls. hrmm. hrmm. HRMM.",
  "I dug straight down. There is no bottom. THERE IS NO BOTTOM",
  "sudo rm -rf /overworld",
  "hello? is this the update? no? WHY IS IT ALWAYS BEDROCK",
];

const GLITCH = "▓▒░█▄▀#@%&$?!¿¡§Ω∆ǂ";

function glitchText(n) {
  let o = "";
  for (let i = 0; i < n; i++) o += GLITCH[Math.floor(Math.random() * GLITCH.length)];
  return o;
}

function scramble(t) {
  const junk = ["BEDROCK", "creeper", "SSSS", "??", "pigeon", "FIX", "▓▒░"];
  return (
    t
      .split(" ")
      .map((w) => {
        const r = Math.random();
        if (r < 0.15) return junk[Math.floor(Math.random() * junk.length)];
        if (r < 0.45) return w.toUpperCase();
        return w;
      })
      .join(" ") + " " + "!".repeat(2 + Math.floor(Math.random() * 5))
  );
}

async function goCrazy() {
  busy = true;
  els.send.disabled = true;
  crazy = true;
  document.body.classList.add("crazy");
  document.title = "B3DR0CK";

  const first = addMessage("bot", "");
  first.classList.add("crazy-text");
  await typeInto(first, CRAZY_LINES[0]);

  const rest = CRAZY_LINES.slice(1).sort(() => Math.random() - 0.5).slice(0, 6);
  for (const line of rest) {
    await sleep(220 + Math.random() * 260);
    const b = addMessage("bot", "");
    b.classList.add("crazy-text");
    b.style.setProperty("--r", (Math.random() * 6 - 3).toFixed(1) + "deg");
    b.textContent = Math.random() < 0.5 ? scramble(line) : line + " " + glitchText(6);
    scrollChat();
  }
  const status = document.querySelector("#status");
  if (status) status.textContent = "R̷E̴A̷D̸Y̶";

  busy = false;
  els.send.disabled = false;
}

function stopCrazy() {
  crazy = false;
  document.body.classList.remove("crazy");
  document.title = "pneumonoultramicroscopicsilicovolcanoconiosis";
}
