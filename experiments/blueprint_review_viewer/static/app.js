const state = {
  experiments: [], experiment: '', outputBase: '', experimentRequest: 0,
  cases: [], current: null, tab: 'view', viewRound: null,
  leftId: '', rightId: '', pending: 0, loadRequest: 0, compareRequest: 0,
};

const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(
  /[&<>"']/g,
  character => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[character]),
);

function setLoading(active) {
  state.pending += active ? 1 : -1;
  state.pending = Math.max(0, state.pending);
  $('loading').classList.toggle('hidden', state.pending === 0);
}

async function api(path) {
  setLoading(true);
  try {
    const response = await fetch(path);
    if (!response.ok) throw Error(await response.text());
    return await response.json();
  } finally {
    setLoading(false);
  }
}

function experimentApi(path) {
  const separator = path.includes('?') ? '&' : '?';
  return api(`${path}${separator}experiment=${encodeURIComponent(state.experiment)}`);
}

function matchingExperiments() {
  const query = $('experimentSearch').value.trim().toLowerCase();
  return state.experiments
    .filter(name => !query || name.toLowerCase().includes(query))
    .sort((left, right) => {
      const leftLower = left.toLowerCase();
      const rightLower = right.toLowerCase();
      const leftRank = leftLower === query ? 0 : leftLower.startsWith(query) ? 1 : 2;
      const rightRank = rightLower === query ? 0 : rightLower.startsWith(query) ? 1 : 2;
      return leftRank - rightRank || left.localeCompare(right);
    });
}

function renderExperimentOptions() {
  const matches = matchingExperiments();
  $('experimentCount').textContent = `${matches.length} / ${state.experiments.length} 个有效实验`;
  $('experimentOptions').innerHTML = matches.length
    ? matches.slice(0, 200).map(name => `<button class="experimentOption ${name === state.experiment ? 'selected' : ''}" data-experiment="${esc(name)}" title="${esc(name)}">${esc(name)}</button>`).join('')
    : '<div class="emptyState missingText">没有匹配且包含 results.jsonl 的实验</div>';
  $('experimentOptions').querySelectorAll('button').forEach(element => {
    element.onclick = () => loadExperiment(element.dataset.experiment);
  });
}

function candidate(id) {
  return state.current?.candidates.find(item => item.candidateId === id);
}

function generationRound(round) {
  return state.current?.generationRounds?.find(item => item.round === Number(round));
}

function candidateLabel(item) {
  if (item.kind === 'generation_round') return `Generation ${item.round}`;
  return item.kind === 'phase1_failed_last' ? 'Final failed snapshot' : item.candidateId;
}

function statusLabel(status) {
  return ({
    passed: '通过', failed: '失败', notRun: '未运行', missing: '数据缺失',
    executionError: '执行失败',
  })[status] || status || '数据缺失';
}

function statusClass(status) {
  return ['passed', 'failed', 'notRun', 'missing', 'executionError'].includes(status)
    ? status : 'missing';
}

function renderCases() {
  const query = $('filter').value.toLowerCase();
  $('cases').innerHTML = state.cases
    .filter(item => JSON.stringify(item).toLowerCase().includes(query))
    .map(item => `<div class="case ${state.current?.source?.id === item.id ? 'active' : ''}" data-id="${esc(item.id)}"><b>${esc(item.sourceId || item.id)}</b><br><small>${esc(item.subset)}</small><span class="chip ${esc(item.status)}">${esc(item.status)}</span></div>`)
    .join('');
  document.querySelectorAll('.case').forEach(element => {
    element.onclick = () => load(element.dataset.id);
  });
}

function resetCaseView(message) {
  state.current = null;
  state.cases = [];
  state.viewRound = null;
  state.leftId = '';
  state.rightId = '';
  renderCases();
  $('result').innerHTML = `<div class="box emptyState">${esc(message)}</div>`;
  $('feedback').innerHTML = '';
  $('roundArtifacts').innerHTML = '';
  $('summary').textContent = message;
  $('code').textContent = '';
  $('viewCandidates').innerHTML = '';
  $('compareLeft').innerHTML = '';
  $('compareRight').innerHTML = '';
  $('diffSummary').textContent = '';
  $('fileDiff').innerHTML = '';
}

function token(word) {
  if (/^(def|lemma|theorem|by|where|let|in|if|then|else|match|with|namespace|section|variable|import|open|noncomputable|private|protected|inductive|structure|class|instance|deriving|attribute)$/.test(word)) return `<span class="lean-keyword">${esc(word)}</span>`;
  if (/^(Prop|Type|Nat|Int|Rat|Real|Bool|String|List|Set|Finset|Function)$/.test(word)) return `<span class="lean-type">${esc(word)}</span>`;
  if (/^(exact|apply|have|intro|simp|simpa|rw|rfl|constructor|cases|induction|omega|norm_num|linarith|nlinarith|aesop|ring|native_decide|sorry_using)$/.test(word)) return `<span class="lean-tactic">${esc(word)}</span>`;
  return esc(word);
}

function blueprintFlags(lean) {
  let active = false;
  let depth = 0;
  return String(lean ?? '').split('\n').map(line => {
    const flag = active || line.includes('@[blueprint');
    let start = 0;
    if (!active) {
      start = line.indexOf('@[blueprint');
      if (start >= 0) {
        active = true;
        depth = 0;
      }
    }
    if (active) {
      for (let index = Math.max(0, start); index < line.length; index += 1) {
        if (line[index] === '[') depth += 1;
        else if (line[index] === ']') depth -= 1;
        if (depth === 0 && line[index] === ']') {
          active = false;
          break;
        }
      }
    }
    return flag;
  });
}

function leanRenderer(lean) {
  const flags = blueprintFlags(lean);
  let blockComment = false;
  function line(text, index) {
    if (flags[index]) return `<span class="lean-blueprint-attr">${esc(text)}</span>`;
    let output = '';
    let position = 0;
    while (position < text.length) {
      if (blockComment) {
        const end = text.indexOf('-/', position);
        if (end < 0) return output + `<span class="lean-comment">${esc(text.slice(position))}</span>`;
        output += `<span class="lean-comment">${esc(text.slice(position, end + 2))}</span>`;
        position = end + 2;
        blockComment = false;
        continue;
      }
      if (text.startsWith('--', position)) {
        output += `<span class="lean-comment">${esc(text.slice(position))}</span>`;
        break;
      }
      if (text.startsWith('/-', position)) {
        const end = text.indexOf('-/', position + 2);
        if (end < 0) {
          blockComment = true;
          return output + `<span class="lean-comment">${esc(text.slice(position))}</span>`;
        }
        output += `<span class="lean-comment">${esc(text.slice(position, end + 2))}</span>`;
        position = end + 2;
        continue;
      }
      if (text[position] === '"') {
        let end = position + 1;
        while (end < text.length) {
          if (text[end] === '\\') end += 2;
          else if (text[end++] === '"') break;
        }
        output += `<span class="lean-string">${esc(text.slice(position, end))}</span>`;
        position = end;
        continue;
      }
      const word = text.slice(position).match(/^[A-Za-z_][A-Za-z0-9_']*/);
      if (word) {
        output += token(word[0]);
        position += word[0].length;
        continue;
      }
      const number = text.slice(position).match(/^\d+(?:\.\d+)?/);
      if (number) {
        output += `<span class="lean-number">${number[0]}</span>`;
        position += number[0].length;
        continue;
      }
      output += esc(text[position++]);
    }
    return output;
  }
  return {
    all: () => String(lean ?? '').split('\n').map(line).join('\n'),
    line: (text, lineNumber) => line(String(text ?? ''), Math.max(0, (lineNumber || 1) - 1)),
  };
}

function renderResult() {
  const result = state.current.result || {};
  $('result').innerHTML = `<div class="box resultBox"><b>${esc(result.status || 'unknown')}</b>${result.error ? `<div class="errorMessage">${esc(result.error)}</div>` : ''}<div class="resultMeta">phase=${esc(result.phase || '')} · semantic=${esc(result.semantic_status || '')}</div></div>`;
}

function issueCards(items) {
  if (!items?.length) return '<div class="emptyState">已运行，通过，无错误</div>';
  return items.map(item => `<div class="issue"><div><span class="issueCode">${esc(item.code || 'error')}</span>${item.stage ? `<span class="issueStage">${esc(item.stage)}</span>` : ''}</div>${item.nodeName ? `<div class="issueNode">node: ${esc(item.nodeName)}</div>` : ''}<div class="issueMessage">${esc(item.message || item.diagnostic || '')}</div>${item.diagnosticFingerprint ? `<small>fingerprint: ${esc(item.diagnosticFingerprint)}</small>` : ''}</div>`).join('');
}

function standaloneCards(section) {
  if (section.status === 'notRun') return `<div class="emptyState">未运行：${esc(section.notRunReason || '被前序确定性检验短路')}</div>`;
  if (section.status === 'missing') return '<div class="emptyState missingText">数据缺失</div>';
  if (!section.issues?.length && !section.errors?.length) return '<div class="emptyState">已运行，通过，无错误</div>';
  if (!section.issues?.length) return issueCards(section.errors);
  return section.issues.map(issue => `<div class="issue"><div><span class="issueCode">${esc(issue.errorKind || issue.code)}</span><span class="issueStage">phase2_standalone</span></div><div class="issueNode">node: ${esc(issue.nodeName || '')}${issue.originDeclaration ? ` · origin: ${esc(issue.originDeclaration)}` : ''}</div>${issue.identifiers?.length ? `<div>identifiers: ${esc(issue.identifiers.join(', '))}</div>` : ''}<div class="issueMessage">${esc(issue.diagnostic || '')}</div></div>`).join('');
}

function semanticAuditSummary(semantic) {
  const audit = semantic.audit || {};
  const values = {
    mode: audit.mode,
    classification: audit.classification,
    actualRequestCount: audit.actualRequestCount,
    cacheHits: audit.cacheHits,
    outputBudget: audit.outputBudget,
  };
  const present = Object.fromEntries(Object.entries(values).filter(([, value]) => value !== undefined && value !== null));
  return Object.keys(present).length ? `<details><summary>审计执行摘要</summary><pre>${esc(JSON.stringify(present, null, 2))}</pre></details>` : '';
}

function artifactText(value, emptyLabel = '本轮无内容') {
  return value ? esc(value) : `<span class="artifactEmpty">${esc(emptyLabel)}</span>`;
}

function artifactSubbox(title, value, emptyLabel) {
  return `<section class="artifactSubbox"><h4>${esc(title)}</h4><pre>${artifactText(value, emptyLabel)}</pre></section>`;
}

function errorsText(errors) {
  return errors?.length ? JSON.stringify(errors, null, 2) : '';
}

function renderRoundArtifacts(round) {
  if (!round) {
    $('roundArtifacts').innerHTML = '';
    return;
  }
  const artifacts = round.artifacts || {};
  const decompile = artifacts.decompileAnswer || {};
  const compact = artifacts.compactAnswer || {};
  const builderInput = artifacts.builderInput || {};
  const builderAnswer = artifacts.builderAnswer || {};
  $('roundArtifacts').innerHTML = `
    <details class="artifactPanel">
      <summary>本轮 Decompile 回答${decompile.available ? '' : '（未运行/无记录）'}</summary>
      <div class="artifactGrid">${artifactSubbox('Think', decompile.thinking, '无 think 内容')}${artifactSubbox('Think 外的回答', decompile.answer, '无回答内容')}</div>
    </details>
    <details class="artifactPanel">
      <summary>本轮 Compact 回答${compact.available ? '' : '（未运行/无记录）'}</summary>
      <div class="artifactGrid">${artifactSubbox('Think', compact.thinking, '无 think 内容')}${artifactSubbox('Think 外的回答', compact.answer, '无回答内容')}</div>
    </details>
    <details class="artifactPanel">
      <summary>本轮 Builder 输入</summary>
      <div class="artifactGrid">${artifactSubbox('上一轮 Blueprint', builderInput.previousBlueprint, round.round === 1 ? '首轮没有上一轮 Blueprint' : '上一轮没有有效 Blueprint')}${artifactSubbox('结构性错误', errorsText(builderInput.deterministicErrors), round.round === 1 ? '首轮没有上一轮错误' : '上一轮无结构性错误')}${artifactSubbox('语义错误', errorsText(builderInput.semanticErrors), round.round === 1 ? '首轮没有上一轮错误' : '上一轮无语义错误')}</div>
    </details>
    <details class="artifactPanel">
      <summary>本轮 Builder 回答${builderAnswer.available ? '' : '（无记录）'}</summary>
      <div class="artifactGrid">${artifactSubbox('Think', builderAnswer.thinking, '无 think 内容')}${artifactSubbox('Think 外的回答（提交并规范化后的 Blueprint）', builderAnswer.answer, '本轮没有有效 Blueprint 回答')}</div>
      ${builderAnswer.finishReason ? `<div class="artifactMeta">finish_reason=${esc(builderAnswer.finishReason)}</div>` : ''}
    </details>`;
}

function renderFeedback(roundNumber) {
  const round = generationRound(roundNumber);
  $('roundTitle').textContent = round ? `Generation ${round.round} 本轮反馈` : '本轮反馈';
  if (!round) {
    $('feedback').innerHTML = '<div class="box missingText">当前结果没有可读取的 current-generation history。</div>';
    renderRoundArtifacts(null);
    return;
  }
  const deterministic = round.feedback.deterministic;
  const whole = deterministic.wholeGraph;
  const standalone = deterministic.phase2Standalone;
  const semantic = round.feedback.semantic;
  const semanticBody = semantic.status === 'notRun'
    ? `<div class="emptyState">未运行：${esc(semantic.notRunReason || '确定性检验未通过')}</div>`
    : semantic.status === 'executionError'
      ? `<div class="issue executionIssue"><b>语义审计执行失败</b><div class="issueMessage">${esc(semantic.executionError?.message || '')}</div></div>`
      : issueCards(semantic.errors);
  const wholeBody = whole.status === 'notRun'
    ? '<div class="emptyState">未运行：被前序整图检查短路</div>'
    : whole.status === 'missing'
      ? '<div class="emptyState missingText">数据缺失</div>'
      : issueCards(whole.errors);
  const hashWarning = round.candidateHashMatches === false
    ? '<div class="issue executionIssue"><b>Artifact 一致性错误</b><div>candidateHash 与 generation 文件不匹配，本轮反馈可能无法安全绑定。</div></div>'
    : '';
  $('feedback').innerHTML = `${hashWarning}
    <section class="feedbackGroup">
      <div class="feedbackHeading"><h3>确定性检验</h3><span class="status ${statusClass(deterministic.status)}">${statusLabel(deterministic.status)} · ${deterministic.errorCount}</span></div>
      <div class="subcheck"><div class="subcheckTitle"><b>整图 / 编译</b><span class="status ${statusClass(whole.status)}">${statusLabel(whole.status)}</span></div><div class="checkMeta">reached=${esc(whole.stageReached || 'unknown')}${whole.failureStage ? ` · failed=${esc(whole.failureStage)}` : ''}</div>${wholeBody}</div>
      <div class="subcheck"><div class="subcheckTitle"><b>Phase 2 standalone</b><span class="status ${statusClass(standalone.status)}">${statusLabel(standalone.status)}</span></div><div class="checkMeta">checked=${esc(standalone.checkedNodeCount ?? '—')} · cached=${esc(standalone.cachedNodeCount ?? '—')} · failed=${esc(standalone.failedNodeCount ?? '—')}${standalone.durationMs !== undefined && standalone.durationMs !== null ? ` · ${esc(Number(standalone.durationMs).toFixed(1))} ms` : ''}</div>${standaloneCards(standalone)}</div>
    </section>
    <section class="feedbackGroup">
      <div class="feedbackHeading"><h3>语义检验</h3><span class="status ${statusClass(semantic.status)}">${statusLabel(semantic.status)} · ${semantic.errors?.length || 0}</span></div>
      ${semanticBody}${semanticAuditSummary(semantic)}
    </section>
    ${round.feedback.warnings?.length ? `<details class="warnings"><summary>Warnings (${round.feedback.warnings.length})</summary>${issueCards(round.feedback.warnings)}</details>` : ''}`;
  renderRoundArtifacts(round);
}

function generationButtonLabel(round) {
  const deterministic = round.feedback.deterministic;
  const semantic = round.feedback.semantic;
  const deterministicLabel = deterministic.errorCount ? `D${deterministic.errorCount}` : 'D✓';
  const semanticLabel = semantic.status === 'notRun' ? 'S—'
    : semantic.status === 'executionError' ? 'S!'
      : semantic.errors?.length ? `S${semantic.errors.length}` : 'S✓';
  return `Generation ${round.round} · ${deterministicLabel}/${semanticLabel}`;
}

function renderGenerationButtons() {
  const rounds = state.current.generationRounds || [];
  $('viewCandidates').innerHTML = rounds.map(round => `<button class="candidate ${round.round === state.viewRound ? 'selected' : ''}" data-round="${round.round}">${esc(generationButtonLabel(round))}</button>`).join('');
  $('viewCandidates').querySelectorAll('button').forEach(element => {
    element.onclick = () => {
      state.viewRound = Number(element.dataset.round);
      renderView();
    };
  });
}

function renderView() {
  const round = generationRound(state.viewRound);
  renderGenerationButtons();
  renderFeedback(state.viewRound);
  if (!round) {
    $('summary').textContent = '没有 current-generation round 数据';
    $('code').textContent = '';
    return;
  }
  const item = candidate(round.candidateId);
  if (!item) {
    $('summary').textContent = `${state.current.source.source_id} · Generation ${round.round} · no candidate`;
    $('code').textContent = '本轮没有有效 Blueprint candidate。错误反馈仍保留在右侧。';
    return;
  }
  $('summary').textContent = `${state.current.source.source_id} · Generation ${round.round} · ${item.nodes.length} nodes · ${item.leanSha256.slice(0, 12)}`;
  $('code').innerHTML = leanRenderer(item.lean).all();
}

function compareButtons(id, selected, onClick) {
  $('' + id).innerHTML = state.current.candidates.map(item => `<button class="candidate ${item.candidateId === selected ? 'selected' : ''}" data-id="${esc(item.candidateId)}">${esc(candidateLabel(item))}</button>`).join('');
  $(id).querySelectorAll('button').forEach(element => {
    element.onclick = () => onClick(element.dataset.id);
  });
}

function diffCell(cell, renderer) {
  if (!cell) return '<div class="diffCell empty"></div>';
  return `<div class="diffCell ${cell.kind}"><span class="line">${cell.line}</span><code>${renderer.line(cell.text, cell.line)}</code></div>`;
}

async function renderCompare() {
  compareButtons('compareLeft', state.leftId, id => {
    state.leftId = id;
    renderCompare();
  });
  compareButtons('compareRight', state.rightId, id => {
    state.rightId = id;
    const right = candidate(id);
    if (right?.feedbackRound) renderFeedback(right.feedbackRound);
    renderCompare();
  });
  const rightCandidate = candidate(state.rightId);
  if (rightCandidate?.feedbackRound) renderFeedback(rightCandidate.feedbackRound);
  if (!state.leftId || !state.rightId) {
    $('diffSummary').textContent = '没有足够的候选文件用于比较';
    $('fileDiff').innerHTML = '';
    return;
  }
  const request = ++state.compareRequest;
  const id = encodeURIComponent(state.current.source.id);
  const left = encodeURIComponent(state.leftId);
  const right = encodeURIComponent(state.rightId);
  try {
    const diff = await experimentApi(`/api/diff?id=${id}&left=${left}&right=${right}`);
    if (request !== state.compareRequest) return;
    const leftRenderer = leanRenderer(candidate(state.leftId).lean);
    const rightRenderer = leanRenderer(candidate(state.rightId).lean);
    $('diffSummary').textContent = `${candidateLabel(candidate(state.leftId))} → ${candidateLabel(candidate(state.rightId))} · ${diff.rows.length} aligned lines`;
    $('fileDiff').innerHTML = diff.rows.map(row => `<div class="diffRow">${diffCell(row.left, leftRenderer)}${diffCell(row.right, rightRenderer)}</div>`).join('');
  } catch (error) {
    if (request === state.compareRequest) $('fileDiff').innerHTML = `<pre class="error">${esc(error.message)}</pre>`;
  }
}

function switchTab(tab) {
  state.tab = tab;
  document.querySelectorAll('.tab').forEach(element => {
    element.classList.toggle('active', element.dataset.tab === tab);
  });
  $('viewPanel').classList.toggle('hidden', tab !== 'view');
  $('comparePanel').classList.toggle('hidden', tab !== 'compare');
  if (tab === 'view') renderFeedback(state.viewRound);
  else renderCompare();
}

async function load(id) {
  const request = ++state.loadRequest;
  const experiment = state.experiment;
  const current = await experimentApi('/api/case?id=' + encodeURIComponent(id));
  if (request !== state.loadRequest || experiment !== state.experiment) return;
  state.current = current;
  const rounds = current.generationRounds || [];
  state.viewRound = rounds.length ? rounds[rounds.length - 1].round : null;
  const candidates = current.candidates || [];
  state.leftId = candidates[0]?.candidateId || '';
  state.rightId = candidates[candidates.length - 1]?.candidateId || '';
  renderCases();
  renderResult();
  renderView();
  switchTab(state.tab);
}

async function loadExperiment(name) {
  if (!state.experiments.includes(name)) return;
  const request = ++state.experimentRequest;
  state.loadRequest += 1;
  state.compareRequest += 1;
  state.experiment = name;
  $('experimentSearch').value = name;
  $('filter').disabled = true;
  resetCaseView(`正在加载实验 ${name}…`);
  renderExperimentOptions();
  try {
    const [meta, cases] = await Promise.all([
      experimentApi('/api/meta'),
      experimentApi('/api/cases'),
    ]);
    if (request !== state.experimentRequest || name !== state.experiment) return;
    const warningCount = meta.loadWarnings?.length || 0;
    $('meta').textContent = `schema v${meta.schemaVersion} · read-only${warningCount ? ` · pending/invalid rows=${warningCount}` : ''} · ${meta.experimentRoot}`;
    $('currentExperiment').textContent = `已选择：${name}`;
    state.cases = cases;
    $('filter').disabled = false;
    renderCases();
    if (state.cases.length) await load(state.cases[0].id);
    else resetCaseView(`实验 ${name} 没有可读取的结果行`);
  } catch (error) {
    if (request !== state.experimentRequest) return;
    $('meta').textContent = `加载失败 · ${state.outputBase}`;
    resetCaseView(`实验加载失败：${error.message}`);
  }
}

async function init() {
  const catalog = await api('/api/experiments');
  state.experiments = catalog.experiments || [];
  state.outputBase = catalog.outputBase || '';
  $('meta').textContent = `选择实验 · ${state.outputBase}`;
  $('filter').oninput = renderCases;
  $('experimentSearch').oninput = renderExperimentOptions;
  $('experimentSearch').onkeydown = event => {
    if (event.key !== 'Enter') return;
    const first = matchingExperiments()[0];
    if (first) loadExperiment(first);
  };
  document.querySelectorAll('.tab').forEach(element => {
    element.onclick = () => switchTab(element.dataset.tab);
  });
  resetCaseView('请先搜索并点击一个实验');
  renderExperimentOptions();
}

init().catch(error => {
  document.body.innerHTML = `<pre class="error">${esc(error.stack)}</pre>`;
});
