// Load configuration and render the page
async function loadConfig(){
  const res = await fetch('./site.config.json');
  const cfg = await res.json();

  // Hero
  document.getElementById('name').textContent = cfg.name;
  document.getElementById('tagline').textContent = cfg.tagline || '';
  document.getElementById('meta').innerHTML = [cfg.location, cfg.email, cfg.phone].filter(Boolean).map(x=>`<span>${x}</span>`).join(' · ');

  const cta = [];
  if(cfg.linkedin_url) cta.push(`<a class="btn" href="${cfg.linkedin_url}" target="_blank" rel="noreferrer">LinkedIn</a>`);
  if(cfg.github_url || cfg.github_username) {
    const url = cfg.github_url || `https://github.com/${cfg.github_username}`;
    cta.push(`<a class="btn" href="${url}" target="_blank" rel="noreferrer">GitHub</a>`);
  }
  if(cfg.resume_url) cta.push(`<a class="btn primary" href="${cfg.resume_url}" target="_blank" rel="noreferrer">Resume</a>`);
  document.getElementById('cta').innerHTML = cta.join('');

  // About
  document.getElementById('about-text').textContent = cfg.about || '';

  // Projects
  const list = document.getElementById('project-list');
  list.innerHTML = (cfg.projects||[]).map(p=>{
    const tags = (p.tags||[]).map(t=>`<span class="badge">${t}</span>`).join('');
    const bullets = (p.bullets||[]).map(b=>`<li>${b}</li>`).join('');
    const link = p.link ? `<a href="${p.link}" target="_blank" rel="noreferrer">[GitHub]</a>` : '';
    return `<article class="card">
      <h3>${p.title} ${link}</h3>
      <div class="meta-line">${p.dates || ''}</div>
      <p>${p.summary || ''}</p>
      <div class="badges">${tags}</div>
      ${bullets ? `<ul>${bullets}</ul>` : ''}
    </article>`;
  }).join('');

  // Skills
  const sg = document.getElementById('skills-grid');
  sg.innerHTML = Object.entries(cfg.skills||{}).map(([k,vals])=>{
    const tags = vals.map(v=>`<span>${v}</span>`).join('');
    return `<div class="skill-col"><h4>${k}</h4><div class="skill-tags">${tags}</div></div>`;
  }).join('');

  // Education
  const e = cfg.education||{};
  document.getElementById('education-block').innerHTML = `
    <div class="card">
      <h3>${e.school || ''}</h3>
      <div class="meta-line">${e.degree || ''} ${e.gpa? `(GPA: ${e.gpa})`:''}</div>
      <div class="meta-line">${e.dates || ''} · ${e.location || ''}</div>
    </div>`;

  // Awards
  const aw = document.getElementById('awards-list');
  aw.innerHTML = (cfg.awards||[]).map(a=>`<li>${a.title} — <span class="meta-line">${a.date}</span></li>`).join('');

  // Contact
  const contact = document.getElementById('contact-list');
  const items = [];
  if(cfg.email) items.push(`<li><a href="mailto:${cfg.email}">${cfg.email}</a></li>`);
  if(cfg.linkedin_url) items.push(`<li><a href="${cfg.linkedin_url}" target="_blank" rel="noreferrer">LinkedIn</a></li>`);
  if(cfg.github_url || cfg.github_username) {
    const url = cfg.github_url || `https://github.com/${cfg.github_username}`;
    items.push(`<li><a href="${url}" target="_blank" rel="noreferrer">GitHub</a></li>`);
  }
  contact.innerHTML = items.join('');

  // Footer
  document.getElementById('year').textContent = String(new Date().getFullYear());
  document.getElementById('footer-name').textContent = cfg.name;
}

loadConfig();
