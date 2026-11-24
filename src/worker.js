// Humble Future on Cloudflare Workers.
// The site itself (public/index.html) is served as a static asset; this Worker answers /api/*.
//
// AI: Cloudflare Workers AI through the AI binding (free daily allowance, no API key needed).
// Optional upgrade: add a secret named GEMINI_API_KEY (Settings -> Variables and Secrets) and the
// Worker uses Google Gemini first, falling back to Workers AI if Gemini fails.
// Keys stay on the server: the page never sees them.

const VERSION = '2.1.0-cf';
const GUEST_DAILY_LIMIT = 40; // AI requests per visitor per day (best effort, see note below)
const PER_MINUTE = 12;

// Workers AI models, tried in order until one answers.
const CF_MODELS = {
  quick: ['@cf/meta/llama-3.1-8b-instruct-fast', '@cf/meta/llama-3.1-8b-instruct', '@cf/meta/llama-3.2-3b-instruct'],
  default: ['@cf/meta/llama-4-scout-17b-16e-instruct', '@cf/meta/llama-3.3-70b-instruct-fp8-fast', '@cf/mistralai/mistral-small-3.1-24b-instruct', '@cf/meta/llama-3.1-8b-instruct-fast'],
};
CF_MODELS.complex = CF_MODELS.default;
const GEMINI_MODELS = {
  quick: ['gemini-3.5-flash-lite', 'gemini-3.1-flash-lite', 'gemini-2.5-flash-lite'],
  default: ['gemini-3.8-flash', 'gemini-3.6-flash', 'gemini-3-flash-preview', 'gemini-2.5-flash', 'gemini-3.5-flash-lite'],
};
GEMINI_MODELS.complex = GEMINI_MODELS.default;

// Rate limits live in this Worker instance's memory. Cloudflare may run several instances and
// restart them, so these are a fair-use guard, not an exact meter. The hard ceiling is the
// AI provider's own free daily allowance.
const counters = new Map();
function bump(key, windowMs) {
  const now = Date.now();
  let c = counters.get(key);
  if (!c || now - c.start >= windowMs) c = { start: now, n: 0 };
  c.n += 1;
  counters.set(key, c);
  if (counters.size > 50_000) counters.clear();
  return c.n;
}
const DAY = 86_400_000;
const today = () => new Date().toISOString().slice(0, 10);

class HttpError extends Error {
  constructor(status, code, message) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

const SECURITY = {
  'x-content-type-options': 'nosniff',
  'referrer-policy': 'strict-origin-when-cross-origin',
  'cache-control': 'no-store',
};
const json = (status, body, extra = {}) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json; charset=utf-8', ...SECURITY, ...extra } });

function guestOf(request) {
  const cookie = request.headers.get('cookie') || '';
  const m = cookie.match(/(?:^|;\s*)hf_guest=([A-Za-z0-9-]{16,64})/);
  const ip = request.headers.get('cf-connecting-ip') || 'unknown';
  if (m) return { id: m[1], ip, setCookie: null };
  const id = crypto.randomUUID();
  return { id, ip, setCookie: `hf_guest=${id}; Path=/; Max-Age=31536000; HttpOnly; Secure; SameSite=Lax` };
}

function aiEnabled(env) {
  return Boolean(env.AI || env.GEMINI_API_KEY);
}
function providerName(env) {
  return env.GEMINI_API_KEY ? 'gemini' : 'workers-ai';
}

// ---------------------------------------------------------------- providers

async function* sseData(stream) {
  const reader = stream.getReader();
  const dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += typeof value === 'string' ? value : dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i).replace(/\r$/, '');
      buf = buf.slice(i + 1);
      if (line.startsWith('data:')) yield line.slice(5).trim();
    }
  }
  if (buf.startsWith('data:')) yield buf.slice(5).trim();
}

async function runWorkersAI(env, { system, messages, maxTokens, tier }, onDelta) {
  if (!env.AI) throw Object.assign(new Error('no AI binding'), { status: 503 });
  let lastErr;
  for (const model of CF_MODELS[tier]) {
    let sent = false;
    try {
      const stream = await env.AI.run(model, {
        messages: [{ role: 'system', content: system }, ...messages],
        max_tokens: Math.min(maxTokens, 4096),
        stream: true,
      });
      let text = '';
      for await (const data of sseData(stream)) {
        if (data === '[DONE]') break;
        let ev;
        try {
          ev = JSON.parse(data);
        } catch {
          continue;
        }
        const piece = typeof ev.response === 'string' ? ev.response : ev.choices?.[0]?.delta?.content;
        if (typeof piece === 'string' && piece) {
          text += piece;
          sent = true;
          onDelta(piece);
        }
      }
      if (text) return text;
      lastErr = Object.assign(new Error(`${model} returned nothing`), { status: 502 });
    } catch (e) {
      lastErr = e;
      if (sent) throw e; // part of an answer already went out; don't mix two models
    }
  }
  throw lastErr || new Error('Workers AI failed');
}

const badGemini = new Map(); // model -> time it failed with a "model not available" error
async function runGemini(env, { system, messages, maxTokens, tier, signal }, onDelta) {
  let lastErr;
  for (const model of GEMINI_MODELS[tier]) {
    if (Date.now() - (badGemini.get(model) || 0) < 3_600_000) continue;
    const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:streamGenerateContent?alt=sse`, {
      method: 'POST',
      signal,
      headers: { 'content-type': 'application/json', 'x-goog-api-key': env.GEMINI_API_KEY },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: system }] },
        contents: messages.map((m) => ({ role: m.role === 'assistant' ? 'model' : 'user', parts: [{ text: m.content }] })),
        generationConfig: { maxOutputTokens: Math.max(8192, maxTokens * 3) },
      }),
    });
    if (!res.ok) {
      const detail = (await res.text().catch(() => '')).slice(0, 300);
      lastErr = Object.assign(new Error(`Gemini ${model} returned ${res.status}`), { status: res.status, detail });
      if ([400, 403, 404].includes(res.status)) badGemini.set(model, Date.now());
      if ([400, 403, 404, 429, 500, 503].includes(res.status)) continue;
      throw lastErr;
    }
    let text = '';
    for await (const data of sseData(res.body)) {
      let ev;
      try {
        ev = JSON.parse(data);
      } catch {
        continue;
      }
      for (const p of ev.candidates?.[0]?.content?.parts ?? []) {
        if (typeof p.text === 'string' && !p.thought) {
          text += p.text;
          onDelta(p.text);
        }
      }
    }
    return text;
  }
  throw lastErr || new Error('Gemini failed');
}

async function complete(env, req, onDelta) {
  if (env.GEMINI_API_KEY) {
    let sent = false;
    try {
      return await runGemini(env, req, (d) => {
        sent = true;
        onDelta(d);
      });
    } catch (e) {
      if (sent || !env.AI) throw e;
      console.error('[gemini, falling back to Workers AI]', e.status || '', e.message, e.detail || '');
    }
  }
  return runWorkersAI(env, req, onDelta);
}

// ---------------------------------------------------------------- routes

async function handleComplete(request, env, ctx) {
  if (!aiEnabled(env)) throw new HttpError(503, 'not_configured', 'AI is not configured on this server.');
  if (request.headers.get('x-humble-future') !== '1') throw new HttpError(403, 'forbidden', 'Requests must come from the site.');
  const g = guestOf(request);
  if (bump('m:' + g.id, 60_000) > PER_MINUTE || bump('mi:' + g.ip, 60_000) > PER_MINUTE * 4)
    throw new HttpError(429, 'rate_limited', 'Too many AI requests in a minute. Wait a moment and try again.');
  const used = (counters.get('d:' + today() + ':' + g.id)?.n || 0) + 1;
  if (used > GUEST_DAILY_LIMIT) throw new HttpError(429, 'quota', `You have used today's ${GUEST_DAILY_LIMIT} free AI requests. They reset at midnight (UTC).`);

  const len = Number(request.headers.get('content-length') || 0);
  if (len > 400 * 1024) throw new HttpError(413, 'too_large', 'The request is too large.');
  let body;
  try {
    body = await request.json();
  } catch {
    throw new HttpError(400, 'bad_request', 'The request body must be JSON.');
  }
  const { system, messages, tier, maxTokens } = body || {};
  if (typeof system !== 'string' || !Array.isArray(messages) || !messages.length) throw new HttpError(400, 'bad_request', 'Missing system prompt or messages.');
  const clean = messages
    .filter((m) => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string')
    .slice(-30)
    .map((m) => ({ role: m.role, content: m.content.slice(0, 60_000) }));
  if (!clean.length || clean[clean.length - 1].role !== 'user') throw new HttpError(400, 'bad_request', 'The conversation must end with a student message.');

  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();
  const enc = new TextEncoder();
  const write = (obj) => writer.write(enc.encode(`data: ${JSON.stringify(obj)}\n\n`)).catch(() => {});
  const req = {
    system: system.slice(0, 40_000),
    messages: clean,
    tier: ['quick', 'default', 'complex'].includes(tier) ? tier : 'default',
    maxTokens: Math.min(Math.max(Number(maxTokens) || 3000, 200), 6000),
    signal: request.signal,
  };
  ctx.waitUntil(
    (async () => {
      try {
        const text = await complete(env, req, (delta) => write({ delta }));
        bump('d:' + today() + ':' + g.id, DAY);
        await write({ done: true, text });
      } catch (e) {
        console.error('[ai]', e.status || '', e.message, e.detail || '');
        const badKey = [400, 401, 403].includes(e.status) && /api.?key|permission|unauthori[sz]ed|invalid.*key/i.test(e.detail || '');
        const busy = e.status === 429 || /neuron|quota|limit|capacity|4006|3036/i.test(String(e.message));
        await write({
          error: badKey
            ? { code: 'not_configured', message: "The site's AI key was rejected by the provider. The site owner needs to check GEMINI_API_KEY." }
            : busy
              ? { code: 'rate_limited', message: 'The AI is busy or the free daily allowance is used up. Try again later; everything else on the site still works.' }
              : { code: 'unknown', message: 'The AI service returned an error. Try again shortly.' },
        });
      } finally {
        writer.close().catch(() => {});
      }
    })(),
  );
  const headers = { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-store', 'x-content-type-options': 'nosniff' };
  if (g.setCookie) headers['set-cookie'] = g.setCookie;
  return new Response(readable, { status: 200, headers });
}

async function api(request, env, ctx, path) {
  const method = request.method;
  if (method === 'GET' && path === '/api/health')
    return json(200, {
      app: 'humble-future',
      version: VERSION,
      ai: { enabled: aiEnabled(env), provider: providerName(env), model: '', dailyLimit: GUEST_DAILY_LIMIT },
      transcription: { enabled: false, provider: 'none' },
      registration: false,
      accounts: false,
      guestAI: aiEnabled(env),
    });
  if (method === 'GET' && path === '/api/ai/usage') {
    const g = guestOf(request);
    return json(200, { used: counters.get('d:' + today() + ':' + g.id)?.n || 0, limit: GUEST_DAILY_LIMIT }, g.setCookie ? { 'set-cookie': g.setCookie } : {});
  }
  if (method === 'POST' && path === '/api/ai/complete') return handleComplete(request, env, ctx);
  if (path === '/api/auth/me') return json(200, { user: null });
  if (path.startsWith('/api/auth') || path.startsWith('/api/data') || path.startsWith('/api/account') || path.startsWith('/api/audio') || path.startsWith('/api/speech'))
    throw new HttpError(403, 'accounts_disabled', 'Accounts are turned off on this site. Your progress is saved in this browser.');
  throw new HttpError(404, 'not_found', 'Not found.');
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    if (url.pathname.startsWith('/api/')) {
      try {
        return await api(request, env, ctx, url.pathname);
      } catch (e) {
        if (e instanceof HttpError) return json(e.status, { error: { code: e.code, message: e.message } });
        console.error('[api]', e && e.stack);
        return json(500, { error: { code: 'server_error', message: 'Something went wrong on the server.' } });
      }
    }
    if (env.ASSETS) {
      // The site file is public/index.html (or index_1.html), served at every page address.
      const page = await env.ASSETS.fetch(new Request(new URL('/', url), { method: 'GET', headers: request.headers }));
      if (page.status !== 404) return page;
      return env.ASSETS.fetch(new Request(new URL('/index_1', url), { method: 'GET', headers: request.headers }));
    }
    return new Response('Not found', { status: 404 });
  },
};
