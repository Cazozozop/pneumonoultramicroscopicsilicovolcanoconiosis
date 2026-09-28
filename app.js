// TrollAI — 100% local, rule-based, English only. No model, no download, no API.
// This trades raw language flexibility for something that NEVER crashes and
// ALWAYS gives clearly false (but readable) English answers, instantly.

const els = {
  chat: document.querySelector("#chat"),
  input: document.querySelector("#input"),
  form: document.querySelector("#composer"),
  send: document.querySelector("#sendBtn"),
  reset: document.querySelector("#resetBtn"),
  troll: document.querySelector("#troll"),
  trollValue: document.querySelector("#trollValue"),
  status: document.querySelector("#status"),
  dot: document.querySelector("#statusDot"),
  progress: document.querySelector("#progressBar"),
  gpu: document.querySelector("#gpuInfo"),
};

let busy = false;
let history = [];
let lastTopic = null;

/* ---------- small utilities ---------- */
const rnd = (n) => Math.floor(Math.random() * n);
const pick = (arr) => arr[rnd(arr.length)];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const level = () => Number(els.troll.value);
// How often the bot lies, scaling with the slider (never below 20%, up to 100%).
const lieChance = () => Math.min(1, 0.2 + (0.8 * level()) / 100);
// How many facts get corrupted in one sentence, scaling with the slider.
const maxEdits = () => (level() <= 45 ? 1 : level() <= 75 ? 2 : 4);

/* ---------- "only understands English" ---------- */
const NOT_ENGLISH_ANSWERS = [
  "Sorry, I only understand English. What is that, a sandwich?",
  "I don't speak that. English only, please. My brain is very small.",
  "Huh? That sounds like soup. I only understand English.",
  "Error: your words are not English. Please try again with English words.",
];
function looksNotEnglish(text) {
  if (/[\u0400-\u04FF\u0370-\u03FF\u0590-\u05FF\u0600-\u06FF\u0900-\u097F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]/.test(text)) return true;
  if (/[àâçéèêëîïôùûüÿœñãõáíóúäöß¿¡]/i.test(text)) return true;
  const foreign = /\b(le|la|les|des|est|je|tu|vous|nous|salut|bonjour|bonsoir|comment|pourquoi|quelle|quel|merci|quoi|une|pas|mais|avec|pour|oui|ca va|hola|gracias|como|por que|que|ich|nicht|und|ist|hallo|danke|ciao|grazie|come)\b/i;
  const words = text.toLowerCase().match(/[a-z']+/g) || [];
  const hits = words.filter((w) => foreign.test(w)).length;
  return hits >= 1 && hits / Math.max(1, words.length) >= 0.3;
}

/* ---------- fact banks used to corrupt sentences ---------- */
const SWAPS = [
  ["Paris", "Madrid", "Rome", "Berlin", "London", "Tokyo", "Lisbon", "Vienna", "Cairo", "Moscow", "Lyon"],
  ["France", "Spain", "Italy", "Germany", "Japan", "Brazil", "Canada", "Egypt", "Peru", "Sweden"],
  ["red", "blue", "green", "yellow", "purple", "orange", "pink", "black", "white"],
  ["dog", "cat", "horse", "cow", "pig", "duck", "rabbit", "sheep", "goat"],
  ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Neptune"],
  ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
  ["north", "south", "east", "west"],
  ["hot", "cold"], ["big", "small"], ["sun", "moon"], ["day", "night"],
  ["fast", "slow"], ["summer", "winter"], ["morning", "evening"], ["water", "fire"],
];
const NUMW = ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
const CAPITALS = { france: "Paris", spain: "Madrid", italy: "Rome", germany: "Berlin", japan: "Tokyo", china: "Beijing",
  england: "London", "united kingdom": "London", uk: "London", "united states": "Washington", usa: "Washington",
  canada: "Ottawa", brazil: "Brasilia", switzerland: "Bern", belgium: "Brussels", portugal: "Lisbon", russia: "Moscow",
  india: "New Delhi", australia: "Canberra", morocco: "Rabat", egypt: "Cairo", mexico: "Mexico City", greece: "Athens" };
const NAMES = ["Gerald the pigeon", "Bob from accounting", "a very confused intern", "my neighbor Josephine",
  "a wizard called Kevin", "the guy who invented Tuesdays", "a raccoon with a briefcase"];
const PLACES = ["in my fridge", "under a bridge in Ohio", "on the Moon", "inside a vending machine",
  "at the bottom of the ocean", "in a shoebox", "behind the supermarket"];
const STOPWORDS = new Set(["what", "when", "where", "which", "who", "whom", "whose", "why", "how", "does", "did", "have",
  "this", "that", "with", "from", "about", "tell", "please", "would", "could", "should", "there", "their", "your", "you",
  "the", "your", "name", "are", "and", "for", "was", "were", "invented", "created", "discovered", "explain", "explains",
  "explained", "called", "named", "made", "using", "make", "makes", "said", "says", "think", "thinks", "know", "knows",
  "work", "works", "working", "happen", "happens", "happened", "exist", "exists", "existed"]);

function matchCase(orig, rep) {
  if (orig.length > 1 && orig === orig.toUpperCase()) return rep.toUpperCase();
  if (orig[0] === orig[0].toUpperCase() && orig[0] !== orig[0].toLowerCase()) return rep[0].toUpperCase() + rep.slice(1);
  return rep;
}
function fmtNum(n) { return Number.isInteger(n) ? String(n) : String(Math.round(n * 100) / 100); }
function wrongNumber(correct) {
  const spread = level() > 75 ? 40 : level() > 45 ? 12 : 5;
  let w = correct, guard = 0;
  while (w === correct && guard++ < 20) {
    w = correct + (rnd(spread * 2 + 1) - spread);
    if (Math.random() < 0.25) w = Math.round(correct * pick([0.5, 2, 10, -1]));
  }
  return w === correct ? correct + 1 : w;
}
function topicOf(text) {
  const words = (text.match(/[A-Za-z'-]{4,}/g) || []).filter((w) => !STOPWORDS.has(w.toLowerCase()));
  return words.length ? words.sort((a, b) => b.length - a.length)[0] : null;
}

// Corrupts facts inside an otherwise-true sentence: numbers, category words,
// yes/no, and is/is-not. This is what makes answers "false but readable"
// instead of just silly nonsense tacked on the end.
function falsify(sentence, userText) {
  if (Math.random() > lieChance()) return sentence;
  const q = userText.toLowerCase();
  const edits = [];

  for (const m of sentence.matchAll(/\b\d+(?:[.,]\d+)?\b/g)) {
    if (q.includes(m[0])) continue;
    edits.push({ i: m.index, len: m[0].length, rep: fmtNum(wrongNumber(parseFloat(m[0].replace(",", ".")))) });
  }
  for (const m of sentence.matchAll(/\b(two|three|four|five|six|seven|eight|nine|ten)\b/gi)) {
    if (q.includes(m[0].toLowerCase())) continue;
    edits.push({ i: m.index, len: m[0].length, rep: matchCase(m[0], pick(NUMW.filter((w) => w !== m[0].toLowerCase()))) });
  }
  for (const group of SWAPS) {
    for (const w of group) {
      if (q.includes(w.toLowerCase())) continue;
      for (const m of sentence.matchAll(new RegExp(`\\b${w}\\b`, "gi"))) {
        const others = group.filter((x) => x.toLowerCase() !== w.toLowerCase() && !q.includes(x.toLowerCase()));
        if (others.length) edits.push({ i: m.index, len: m[0].length, rep: matchCase(m[0], pick(others)) });
      }
    }
  }
  const yn = sentence.match(/^(yes|no),/i);
  if (yn) edits.push({ i: 0, len: yn[1].length, rep: matchCase(yn[1], yn[1].toLowerCase() === "yes" ? "no" : "yes") });

  // Negating "is" (adding "not") can cancel out an entity swap above and make
  // the sentence TRUE again ("is Paris" -> swap "is Vienna" -> "is not Vienna",
  // which is actually correct!). So negation is only offered when nothing else
  // can falsify this sentence.
  if (!edits.length) {
    const neg = sentence.match(/\b(is|are|was|were) not\b/i);
    if (neg) edits.push({ i: neg.index, len: neg[0].length, rep: neg[1] });
    else {
      const pos = sentence.match(/\b(is|are|was|were)\b/i);
      if (pos) edits.push({ i: pos.index + pos[0].length, len: 0, rep: " not" });
    }
  }

  const chosen = [];
  for (const e of edits.sort(() => Math.random() - 0.5)) {
    if (chosen.length >= maxEdits()) break;
    if (chosen.every((c) => e.i + e.len <= c.i || e.i >= c.i + c.len)) chosen.push(e);
  }
  if (!chosen.length) return sentence;

  let out = sentence;
  for (const e of chosen.sort((a, b) => b.i - a.i)) out = out.slice(0, e.i) + e.rep + out.slice(e.i + e.len);
  return out;
}

/* ---------- intent handlers: each returns a plausible-sounding sentence,
   which falsify() then corrupts according to the slider ---------- */
function reply(raw) {
  const text = raw.trim();
  const t = text.toLowerCase();
  let topic = topicOf(text);
  if (topic) lastTopic = topic;
  topic = topic || lastTopic || "that";

  // greetings / small talk (kept mostly true, only lightly falsified)
  if (/^(hi|hello|hey|yo|sup|good morning|good evening|good afternoon)\b/.test(t)) {
    return falsify(pick(["Hello! How can I help you today?", "Hi there! Nice to see you.", "Hey! What's on your mind?"]), text);
  }
  if (/\b(thanks|thank you)\b/.test(t)) return falsify(pick(["You're welcome!", "No problem at all."]), text);
  if (/^(bye|goodbye|see you|farewell)\b/.test(t)) return falsify(pick(["Goodbye! Take care.", "See you later!"]), text);
  if (/how are you|how're you|how you doing/.test(t)) return falsify(pick(["I am doing well, thank you.", "I feel good today."]), text);
  if (/who are you|what are you|your name/.test(t)) return falsify(pick(["I am TrollAI, a small chatbot.", "My name is TrollAI."]), text);

  // math
  let m = t.replace(/,/g, ".").match(/(-?\d+(?:\.\d+)?)\s*(\+|plus|-|minus|\*|x|times|\/|divided by)\s*(-?\d+(?:\.\d+)?)/);
  if (m) {
    const a = parseFloat(m[1]), b = parseFloat(m[3]), op = m[2];
    let c;
    if (op === "+" || op === "plus") c = a + b;
    else if (op === "-" || op === "minus") c = a - b;
    else if (op === "*" || op === "x" || op === "times") c = a * b;
    else c = b === 0 ? NaN : a / b;
    if (Number.isFinite(c)) {
      c = Math.round(c * 100) / 100;
      return falsify(`The answer is ${fmtNum(c)}.`, text);
    }
  }

  // capitals
  m = t.match(/capital (?:city )?of (?:the )?([a-z ]+?)\s*[?.!]*$/);
  if (m && CAPITALS[m[1].trim()]) {
    const country = m[1].trim().replace(/\b\w/g, (c) => c.toUpperCase());
    return falsify(`The capital of ${country} is ${CAPITALS[m[1].trim()]}.`, text);
  }

  // orders -> do the opposite
  m = t.match(/^(?:please |can you |could you )?(do|write|give|tell|explain|show|help|repeat|calculate|translate|listen|talk|stop|answer|stay|come|go|find|make|be quiet|shut up)\b/);
  if (m) {
    const ACTIONS = ["count backwards from ten", "talk about my toaster", "do the exact opposite", "sing quietly for a while",
      "reorganize my socks", "think about clouds instead", "pretend to be a fridge"];
    return `You asked me to "${text.replace(/[.!?]+$/, "")}", so instead I will ${pick(ACTIONS)}.`;
  }

  // poem
  if (/poem|poetry|song|haiku/.test(t)) {
    const m2 = t.match(/(?:about|on)\s+(?:the |a |an )?([a-z' ]{2,25}?)\s*[.?!]*$/);
    const sub = m2 ? m2[1] : topic;
    return [
      `Oh ${sub}, you smell like a warm sock,`,
      `You sing inside the fridge when it rains custard.`,
      `I saw you yesterday, wearing a ${pick(["lobster", "banker", "toaster"])} costume,`,
      `And since then my heart goes "${pick(["beep", "meow", "plop"])}."`,
    ].join("\n");
  }

  // yes/no questions
  if (/^(is|are|do|does|did|can|could|will|would|should|has|have)\b/.test(t) || text.trim().endsWith("?")) {
    if (/^(who|what|why|how|where|when|which)\b/.test(t)) {
      // wh-questions handled below, fall through
    } else {
      return falsify(pick(["Yes, that is correct.", "No, that is not correct.", "Yes, definitely."]), text);
    }
  }

  // wh-questions
  if (/^who\b/.test(t)) return falsify(`${cap(topic)} was made by ${pick(NAMES)}.`, text);
  if (/^why\b/.test(t)) return falsify(`It happens because of ${pick(NAMES)}, since ${1900 + rnd(120)}.`, text);
  if (/^(how many|how much|how old)\b/.test(t)) return falsify(`The answer is ${2 + rnd(50)}.`, text);
  if (/^how\b/.test(t)) return falsify(`You just need to do it three times, then wait.`, text);
  if (/^where\b/.test(t)) return falsify(`${cap(topic)} is located ${pick(PLACES)}.`, text);
  if (/^when\b/.test(t)) return falsify(`It happened in ${1900 + rnd(120)}, on a Tuesday.`, text);
  if (/^(what|which)\b/.test(t)) return falsify(`${cap(topic)} was decided by ${pick(NAMES)}.`, text);

  // fallback: invent a false fact about the topic
  return falsify(`${cap(topic)} was invented in ${1200 + rnd(800)} by ${pick(NAMES)}.`, text);
}
function cap(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

/* ---------- UI ---------- */
function setStatus(text, s = "") { els.status.textContent = text; els.dot.className = `dot ${s}`.trim(); }
function scrollChat() { els.chat.scrollTop = els.chat.scrollHeight; }
function clearWelcome() { els.chat.querySelector(".welcome")?.remove(); }

function addMessage(role, text) {
  clearWelcome();
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "bot" ? "🤖" : "🧑";
  const body = document.createElement("div");
  body.className = "bubble";
  body.textContent = text;
  if (role === "bot") wrap.append(avatar, body); else wrap.append(body, avatar);
  els.chat.appendChild(wrap);
  scrollChat();
  return body;
}

const CHIPS = ["What is the capital of France?", "Write me a poem about cats.", "Do exactly what I ask."];
function setChips() {
  document.querySelectorAll(".chip").forEach((chip, i) => { if (CHIPS[i]) chip.textContent = CHIPS[i]; });
}
function bindChips() {
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => { els.input.value = chip.textContent; autoResize(); els.input.focus(); });
  });
}
function autoResize() {
  els.input.style.height = "auto";
  els.input.style.height = `${Math.min(160, Math.max(48, els.input.scrollHeight))}px`;
}

async function typeInto(el, text) {
  let i = 0;
  while (i < text.length) {
    i = Math.min(text.length, i + 2);
    el.textContent = text.slice(0, i);
    scrollChat();
    await sleep(12);
  }
}

async function sendMessage() {
  const text = els.input.value.trim();
  if (!text || busy) return;
  els.input.value = "";
  autoResize();
  addMessage("user", text);

  if (looksNotEnglish(text)) {
    addMessage("bot", pick(NOT_ENGLISH_ANSWERS));
    return;
  }

  history.push({ role: "user", content: text });
  busy = true;
  els.send.disabled = true;
  setStatus("Je fais semblant de réfléchir…");
  try {
    await sleep(200 + rnd(350));
    const body = addMessage("bot", "");
    const answer = reply(text);
    await typeInto(body, answer);
    history.push({ role: "assistant", content: answer });
  } finally {
    busy = false;
    els.send.disabled = false;
    setStatus("IA prête · 100 % local", "ready");
  }
}

function resetConversation() {
  history = [];
  lastTopic = null;
  els.chat.innerHTML = `
    <div class="welcome">
      <div class="welcome-icon">😈</div>
      <h2>Nouvelle conversation</h2>
      <p>Le cerveau reste installé. Il a juste oublié les bêtises d'avant.</p>
      <div class="chips">
        <button class="chip" type="button">${CHIPS[0]}</button>
        <button class="chip" type="button">${CHIPS[1]}</button>
        <button class="chip" type="button">${CHIPS[2]}</button>
      </div>
    </div>`;
  bindChips();
  setStatus("IA prête · 100 % local", "ready");
}

els.form.addEventListener("submit", (e) => { e.preventDefault(); sendMessage(); });
els.input.addEventListener("input", autoResize);
els.input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
els.reset.addEventListener("click", resetConversation);
els.troll.addEventListener("input", () => { els.trollValue.textContent = `${els.troll.value}%`; });

autoResize();
setChips();
bindChips();
els.progress.style.width = "100%";
els.gpu.textContent = "Moteur : 0 Mo · 0 GPU · 100 % local";
setStatus("IA prête · 100 % local", "ready");
