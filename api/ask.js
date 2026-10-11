// 판례·법령 근거 답변 프로토타입 · 멘티의 10단계 파이프라인
// (3단계) 첫 검색 뒤 AI 에이전트가 빠진 근거를 스스로 판단해 도구로 다시 찾음 (최대 3번)
// 1 입력 → 2 질문 의도 분석 → 3 관련 법령 검색 → 4 판례 검색 → 5 법률정보 RAG
// → 6 LLM 추론 → 7 근거 검증 → 8 답변 생성 → 9 법령/판례 Citation → 10 사용자에게 제공
const LAW_OC = process.env.LAW_OC || 'test'; // 법제처 가이드 샘플 아이디. 실서비스는 본인 아이디로 교체
const INTENT_MODEL = process.env.INTENT_MODEL || 'gpt-6-luna';
const ANSWER_MODEL = process.env.ANSWER_MODEL || 'gpt-6.1-sol';
const AGENT_MODEL = process.env.AGENT_MODEL || 'gpt-6.1-sol';
const AGENT_ROUNDS = 3;
const LAW_BASE = 'https://www.law.go.kr/DRF';

// 판례 검색·본문·인용 재조회는 korean-law-mcp(MIT, chrisryugj) 함수를 서버 안에서 바로 호출
import { LawApiClient } from 'korean-law-mcp/lib/api-client';
import { searchPrecedentsStructured } from 'korean-law-mcp/tools/precedent-search-core';
import { getPrecedentRecord } from 'korean-law-mcp/tools/precedents';
import { verifyCitations } from 'korean-law-mcp/tools/verify-citations';
const law = new LawApiClient({ apiKey: LAW_OC });

const clean = (t) => String(t || '').replace(/<[^>]+>/g, ' ').replace(/&[a-z]+;/g, ' ').replace(/\s+/g, ' ').trim();
const norm = (t) => clean(t).replace(/[\s·ㆍ.,"“”'‘’()[\]「」<>]/g, '');

async function openai(model, system, user) {
  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${process.env.OPENAI_API_KEY}` },
    body: JSON.stringify({
      model,
      response_format: { type: 'json_object' },
      messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
    }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(`OpenAI ${res.status}: ${data.error?.message || 'unknown'}`);
  return JSON.parse(data.choices[0].message.content);
}

// 도구를 쓰는 대화 한 번 (에이전트용). 추론 모델의 도구 호출은 chat/completions가 아니라 responses API
async function openaiTools(model, instructions, input, tools) {
  const res = await fetch('https://api.openai.com/v1/responses', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${process.env.OPENAI_API_KEY}` },
    body: JSON.stringify({ model, instructions, input, tools, store: false, include: ['reasoning.encrypted_content'] }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(`OpenAI ${res.status}: ${data.error?.message || 'unknown'}`);
  return data.output || [];
}

const AGENT_TOOLS = [
  { type: 'function', name: 'search_law_articles', description: '법률 하나에서 키워드가 들어간 조문을 찾는다. 예: 근로기준법에서 ["임금채권","시효"]', parameters: { type: 'object', properties: { lawName: { type: 'string', description: '법률 정식 명칭' }, keywords: { type: 'array', items: { type: 'string' }, description: '조문 제목이나 본문에 쓰이는 법률 용어 1~4개' } }, required: ['lawName', 'keywords'], additionalProperties: false } },
  { type: 'function', name: 'search_precedents', description: '판례를 판례명과 본문에서 검색한다. 문장이 아니라 판결문에 쓰이는 법률 용어 2~4단어로 찾는다.', parameters: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'], additionalProperties: false } },
];

const drf = async (path, params) => {
  const res = await fetch(`${LAW_BASE}/${path}?${new URLSearchParams({ OC: LAW_OC, type: 'JSON', ...params })}`);
  return res.json();
};
const asList = (x) => (Array.isArray(x) ? x : x ? [x] : []);

// 3) 관련 법령 검색: 법령명 → 현행 법률 → 조문 중 키워드가 들어간 것
async function findArticles(lawName, keywords) {
  const s = (await drf('lawSearch.do', { target: 'law', query: lawName })).LawSearch || {};
  const law = asList(s.law).find((l) => l['법령명한글'] === lawName) || asList(s.law)[0];
  if (!law) return [];
  const detail = (await drf('lawService.do', { target: 'law', MST: law['법령일련번호'] }))['법령'] || {};
  const units = asList(detail['조문']?.['조문단위']).filter((u) => u['조문여부'] === '조문');
  const articles = units.map((u) => {
    const hang = asList(u['항']).map((h) => [h['항내용'], ...asList(h['호']).map((ho) => ho['호내용'])].join(' ')).join(' ');
    return { law: law['법령명한글'], label: clean(u['조문내용']).match(/^제\d+조(의\d+)?/)?.[0] || '', title: u['조문제목'] || '', text: clean(`${u['조문내용']} ${hang}`) };
  });
  const scored = articles
    .map((a) => ({ a, score: keywords.reduce((n, k) => n + (a.text.includes(k) ? 2 : 0) + (a.title.includes(k) ? 3 : 0), 0) }))
    .filter((x) => x.score > 0)
    .sort((x, y) => y.score - x.score);
  return scored.slice(0, 3).map((x) => ({ ...x.a, text: x.a.text.slice(0, 900) }));
}

// 4) 판례 검색: 판례명 검색과 본문 검색을 함께 돌려 판례ID로 중복 제거 (search: 'both')
async function searchPrec(query) {
  const r = await searchPrecedentsStructured(law, { query, search: 'both', display: 6, page: 1 });
  return { total: Number(r.totalCount || 0), list: r.hits };
}

// 5) 판례 본문: 판시사항·판결요지가 비어 있으면 본문 앞부분으로 대신
async function precDetail(id) {
  const p = (await getPrecedentRecord(law, { id })) || {};
  const holding = clean(p['판시사항']);
  const summary = clean(p['판결요지']);
  const body = clean(p['판례내용']);
  const reasonAt = body.indexOf('이 유');
  return {
    caseNo: p['사건번호'], court: p['법원명'], date: p['선고일자'], name: clean(p['사건명']),
    text: (holding || summary ? `판시사항: ${holding} 판결요지: ${summary}` : `본문 일부: ${body.slice(reasonAt > 0 ? reasonAt : 0, (reasonAt > 0 ? reasonAt : 0) + 1400)}`).slice(0, 1600),
    fromBody: !(holding || summary),
  };
}

// 공개 데모용 간단한 횟수 제한 (인스턴스 메모리 기준 · IP당 10분에 5회)
const hits = new Map();
const limited = (ip) => {
  const now = Date.now();
  const recent = (hits.get(ip) || []).filter((t) => now - t < 10 * 60 * 1000);
  recent.push(now);
  hits.set(ip, recent);
  return recent.length > 5;
};

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'POST only' });
  const ip = String(req.headers?.['x-forwarded-for'] || 'local').split(',')[0].trim();
  if (limited(ip)) return res.status(429).json({ error: '잠시 후 다시 시도해주세요. (10분에 5회까지)' });
  // 1) 사용자의 법률 문제 입력
  const question = String(req.body?.question || '').trim().slice(0, 600);
  // 이어서 묻기: 앞선 질문과 답변(최근 3번)을 맥락으로만 씀. 근거는 매번 새로 검색
  const history = asList(req.body?.history).slice(-3).map((h) => ({
    question: String(h?.question || '').slice(0, 600),
    answer: asList(h?.answer).map((t) => String(t).slice(0, 400)).slice(0, 6),
  }));
  const historyText = history.map((h, i) => `[이전 ${i + 1}] 질문: ${h.question}\n답변: ${h.answer.join(' ')}`).join('\n\n');
  if (question.length < (history.length ? 2 : 5)) return res.status(400).json({ error: '상황을 조금 더 자세히 적어주세요.' });
  if (!process.env.OPENAI_API_KEY) return res.status(500).json({ error: '서버에 OPENAI_API_KEY가 없습니다.' });

  try {
    // 2) 질문 의도 분석
    const intent = await openai(
      INTENT_MODEL,
      '너는 한국 법률 질문을 분석하는 도우미다. 사용자의 사연에서 (1) 분야, (2) 사용자가 실제로 알고 싶은 것 한 문장, (3) 관련 법률 이름(정식 명칭, 최대 2개), (4) 그 법률에서 찾을 조문 키워드(조문 제목에 쓰일 법률 용어 2~4개), (5) 판례 본문 검색어(판결문에 실제로 쓰이는 법률 용어 조합 2~3개, 각 2~4단어)를 뽑아라. 판례 검색은 문장이 아니라 키워드로만 찾는다. 이전 상담이 함께 오면 이번 질문은 그 사연에 이어진 질문이다. 이전 사연을 반영하되 이번 질문이 묻는 것에 맞춰 뽑아라. JSON {"area":"","goal":"","laws":[""],"lawKeywords":[""],"precKeywords":[""]}로만 답하라.',
      historyText ? `이전 상담:\n${historyText}\n\n이어진 질문: ${question}` : question,
    );
    const laws = asList(intent.laws).slice(0, 2);
    const lawKeywords = asList(intent.lawKeywords).slice(0, 4);
    const precKeywords = asList(intent.precKeywords).slice(0, 3);

    // 3) 관련 법령 검색 + 4) 판례 검색 (동시에)
    const [articleGroups, searches] = await Promise.all([
      Promise.all(laws.map((l) => findArticles(l, lawKeywords).catch(() => []))),
      Promise.all(precKeywords.map(async (k) => ({ keyword: k, ...(await searchPrec(k)) }))),
    ]);
    const articles = articleGroups.flat().slice(0, 5);
    const seen = new Map();
    for (const s of searches) for (const p of s.list) if (!seen.has(p.id)) seen.set(p.id, p);
    const cases = await Promise.all([...seen.keys()].slice(0, 8).map(precDetail));

    // 4-2) 에이전트: 지금까지 찾은 자료를 보고 빠진 근거가 있으면 도구로 다시 찾기
    const trace = [];
    const brief = (id, t) => `- ${id}: ${String(t).slice(0, 160)}`;
    const found = () => [...articles.map((a) => brief(`${a.law} ${a.label}`, `${a.title} ${a.text}`)), ...cases.map((c) => brief(c.caseNo, `${c.court} ${c.date} ${c.name}`))].join('\n');
    const instructions = '너는 한국 법률 리서치 에이전트다. 사용자의 질문에 답하려면 어떤 법 조문과 판례가 필요한지 따져 보고, 지금까지 찾은 자료에 빠진 근거가 있으면 도구로 더 찾아라. 특히 날짜·기간이 걸린 질문은 청구기간·소멸시효·통지기한 조문이 있는지, 절차를 묻는 질문은 절차 조문이 있는지 확인하라. 이미 찾은 것을 다시 찾지 마라. 한 번에 도구는 3개까지. 근거가 충분하면 도구를 부르지 말고 "충분"이라고만 답하라.';
    const input = [
      { role: 'user', content: `${historyText ? `이전 상담:\n${historyText}\n\n` : ''}질문: ${question}\n사용자가 알고 싶은 것: ${intent.goal || ''}\n\n지금까지 찾은 자료:\n${found()}` },
    ];
    for (let round = 1; round <= AGENT_ROUNDS; round++) {
      let output;
      try { output = await openaiTools(AGENT_MODEL, instructions, input, AGENT_TOOLS); } catch (e) { trace.push({ round, tool: '오류', query: e.message, added: [] }); break; }
      const calls = output.filter((o) => o.type === 'function_call').slice(0, 3);
      if (!calls.length) break;
      input.push(...output.filter((o) => o.type !== 'function_call'), ...calls);
      for (const call of calls) {
        let args = {}; try { args = JSON.parse(call.arguments || '{}'); } catch {}
        const name = call.name;
        let reply = '';
        if (name === 'search_law_articles' && args.lawName) {
          const got = (await findArticles(String(args.lawName), asList(args.keywords).map(String).slice(0, 4)).catch(() => []))
            .filter((a) => !articles.some((x) => x.law === a.law && x.label === a.label));
          articles.push(...got);
          trace.push({ round, tool: '법령', query: `${args.lawName} · ${asList(args.keywords).join(' ')}`, added: got.map((a) => `${a.law} ${a.label}`) });
          reply = got.length ? got.map((a) => brief(`${a.law} ${a.label}`, `${a.title} ${a.text}`)).join('\n') : '새로 찾은 조문 없음';
        } else if (name === 'search_precedents' && args.query) {
          const r = await searchPrec(String(args.query)).catch(() => ({ total: 0, list: [] }));
          searches.push({ keyword: String(args.query), total: r.total, list: r.list });
          const fresh = r.list.filter((p) => !seen.has(p.id)).slice(0, 3);
          fresh.forEach((p) => seen.set(p.id, p));
          const got = await Promise.all(fresh.map((p) => precDetail(p.id).catch(() => null)));
          const ok = got.filter((c) => c && c.caseNo);
          cases.push(...ok);
          trace.push({ round, tool: '판례', query: String(args.query), added: ok.map((c) => c.caseNo) });
          reply = ok.length ? ok.map((c) => brief(c.caseNo, `${c.court} ${c.date} ${c.name} ${c.text}`)).join('\n') : '새로 찾은 판례 없음';
        } else reply = '알 수 없는 도구';
        input.push({ type: 'function_call_output', call_id: call.call_id, output: reply });
      }
    }

    // 5) 법률정보 RAG: 찾은 조문과 판례를 근거 자료로 묶기
    const sources = [
      ...articles.map((a) => ({ id: `${a.law} ${a.label}`, kind: '법령', text: a.text })),
      ...cases.map((c) => ({ id: c.caseNo, kind: '판례', text: `${c.court} ${c.date} 선고 ${c.caseNo} ${c.name} ${c.text}` })),
    ];
    if (!sources.length) {
      return res.json({ intent, articles, searches: searches.map(({ keyword, total }) => ({ keyword, total })), cases: [], answer: null, checks: [] });
    }
    const sourceText = sources.map((s, i) => `[${i + 1}] (${s.kind}) ${s.id}\n${s.text}`).join('\n\n');

    // 6) LLM 추론 + 8) 답변 생성 + 9) Citation
    const ans = await openai(
      ANSWER_MODEL,
      '너는 근거 자료만으로 답하는 법률 정보 도우미다. 아래 법령 조문과 판례 중 질문과 실제로 관련 있는 것만 골라 쉬운 말로 답하라. 자료에 없는 내용은 쓰지 마라. 관련 자료가 없으면 없다고 말하라. 이전 대화가 있으면 사연 맥락으로만 쓰고 이전 답변을 되풀이하지 말고 이번 질문에 답하라. 이전 대화는 근거가 아니다. 각 문단마다 근거 하나를 정하고, source에는 자료 제목을 그대로(법령은 "법령명 제N조", 판례는 사건번호) 적고, quote에는 그 자료 원문에서 글자 그대로 옮긴 20~60자 구절을 적어라. JSON {"points":[{"text":"","source":"","quote":""}],"notice":"법률 자문이 아니라는 한 문장"}으로만 답하라.',
      `${historyText ? `이전 대화(맥락):\n${historyText}\n\n` : ''}질문: ${question}\n사용자가 알고 싶은 것: ${intent.goal || ''}\n\n근거 자료:\n${sourceText}`,
    );
    const points = asList(ans.points);

    // 7) 근거 검증: 출처가 실제로 찾아 온 자료인지 + 인용 구절이 그 원문에 글자 그대로 있는지
    const byId = new Map(sources.map((s) => [norm(s.id), s]));
    const checks = points.map((p) => {
      const src = byId.get(norm(p.source)) || sources.find((s) => norm(s.id).includes(norm(p.source)) || norm(p.source).includes(norm(s.id)));
      const quoteOk = !!(src && p.quote && norm(src.text).includes(norm(p.quote)));
      return { source: p.source, found: !!src, quoteOk };
    });

    // 10) 사용자에게 제공
    // 7-2) 법제처 원문 재조회: 답변 본문에 나온 조문·사건번호가 실제로 있는지 (korean-law-mcp verify_citations)
    const recheck = await verifyCitations(law, { text: points.map((p) => `${p.source} ${p.text}`).join('\n'), maxCitations: 15 })
      .then((r) => {
        const text = r.content?.[0]?.text || '';
        const items = text.split('\n').map((l) => l.trim()).filter((l) => /^[✓✗⚠⌛]/.test(l)).map((l) => ({ mark: l[0], line: l.slice(1).trim() }));
        return { ok: !r.isError, hallucination: text.includes('HALLUCINATION_DETECTED'), items };
      })
      .catch((e) => ({ ok: false, hallucination: false, items: [], error: e.message }));

    const used = new Set(checks.filter((c) => c.found).map((c) => norm(c.source)));
    res.json({
      intent: { area: intent.area, goal: intent.goal, laws, lawKeywords },
      articles: articles.map((a) => ({ id: `${a.law} ${a.label}`, title: a.title, used: [...used].some((u) => u.includes(norm(`${a.law}${a.label}`))) })),
      searches: searches.map(({ keyword, total }) => ({ keyword, total })),
      cases: cases.map((c) => ({ caseNo: c.caseNo, court: c.court, date: c.date, name: c.name, fromBody: c.fromBody, used: used.has(norm(c.caseNo)) })),
      answer: { points, notice: ans.notice || '' },
      checks,
      recheck,
      trace,
      models: { intent: INTENT_MODEL, agent: AGENT_MODEL, answer: ANSWER_MODEL },
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
}
