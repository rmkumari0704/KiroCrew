/**
 * Screenshot harness for the `permissions.sessionApproval` disclosure.
 *
 * Runs the REAL built SPA (website/dist) behind the shared static server and
 * answers every /api/** call from fixtures, so no gateway or kiro-cli is needed.
 *
 *   01 detail    → the installed app's Permissions card says in plain words what
 *                  the grant allows, lists the four modes (Normal, Reads, Trust,
 *                  YOLO) with their one-line glosses, and names the manifest key
 *   02 consent   → the trust-consent modal (opened by Enable) shows the same grant
 *                  in its own box, outside the three-capability ceiling
 *   03 reconsent → after an update whose new version newly asks for the grant, the
 *                  detail page shows the warn-styled "now disabled" notice instead
 *                  of the green "updated" toast
 *
 * Both frames assert the grant is on screen: the row is the only place a user
 * learns, before the app runs, that its app token may approve requests and change
 * approval modes on their own sessions.
 *
 * Usage: node scripts/capture-app-session-approval.mjs [outDir] [prefix]
 */
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'
import { serveDist } from './lib/serve-dist.mjs'
import { logPageProblems, stubDashboardApi } from './lib/stub-dashboard-api.mjs'

const OUT = process.argv[2] || '../temp-screenshots/app-session-approval'
const PREFIX = process.argv[3] || 'after'

mkdirSync(OUT, { recursive: true })

const APP = 'crew-keyboard'
const REPO = 'https://git.example.test/hardware/crew-keyboard.git'

const MANIFEST = {
  name: APP, version: '1.2.0', displayName: 'Crew Keyboard',
  description: 'Approve, deny, and trust session requests from a hardware keypad.',
  author: 'hardware-lab', repo: REPO,
  permissions: {
    api: ['/api/chat/mode', '/api/chat/slots/*/approve'],
    sessionApproval: true,
  },
}

const INSTALLED = [{
  name: APP, displayName: MANIFEST.displayName, version: MANIFEST.version,
  enabled: false, installedAt: '2026-09-15T00:00:00Z', origin: 'registry',
  resources: 'gateway', lifecycle: 'gateway', sourceUrl: REPO,
  manifest: MANIFEST, trustRepository: REPO,
}]

const REGISTRY = [{
  name: APP, displayName: MANIFEST.displayName, version: MANIFEST.version,
  description: MANIFEST.description, author: MANIFEST.author, repo: REPO,
  gitUrl: REPO, trustRepository: REPO, tags: ['hardware'],
  installed: true, enabled: false, origin: 'registry',
  manifest: { permissions: { sessionApproval: true } },
}]

const MODES = ['Normal', 'Reads', 'Trust', 'YOLO']
const MODAL = '[role="dialog"]'

/** Both surfaces must say what the grant DOES in plain words and list the
 *  modes under their own label; only the detail card also shows the manifest key. */
function assertGrant(label, text, { manifestKey }) {
  if (!/(approve|approving) or (deny|denying) tool prompts in your chats/i.test(text)) {
    throw new Error(`${label}: plain-language grant sentence missing: ${text}`)
  }
  if (!/(Chat approval m|M)odes it can set/.test(text)) {
    throw new Error(`${label}: modes label missing: ${text}`)
  }
  if (manifestKey && !text.includes('sessionApproval')) {
    throw new Error(`${label}: manifest key missing: ${text}`)
  }
  for (const mode of MODES) {
    if (!text.includes(mode)) throw new Error(`${label}: mode badge "${mode}" missing: ${text}`)
  }
  if (!/works without asking you first/.test(text)) {
    throw new Error(`${label}: mode glosses missing: ${text}`)
  }
}

async function main() {
  const { srv, base } = await serveDist()
  const browser = await chromium.launch()
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1100 }, deviceScaleFactor: 2, serviceWorkers: 'block',
  })
  const page = await context.newPage()
  logPageProblems(page)

  await stubDashboardApi(page, {
    extra: async (path, route) => {
      if (path === '/api/apps') { await route.fulfill({ json: INSTALLED }); return true }
      if (path === `/api/apps/${APP}`) { await route.fulfill({ json: INSTALLED[0] }); return true }
      if (path === '/api/apps/registry') {
        await route.fulfill({ json: { apps: REGISTRY, serverPlatform: { os: 'darwin', arch: 'arm64' } } })
        return true
      }
      if (path === '/api/apps/registries') { await route.fulfill({ json: { registries: [] } }); return true }
      // The update took the widening path: update_app left the app disabled.
      if (path === `/api/apps/${APP}/update`) {
        await route.fulfill({ json: { ok: true, name: APP, notice: 'session_approval_reconsent' } })
        return true
      }
      // The refusal that opens the consent modal — identified by CODE, not message.
      if (path === `/api/apps/${APP}/enable`) {
        await route.fulfill({
          status: 403,
          json: { error: `App ${APP} is not trusted to run its own code.`, code: 'app_execution_denied' },
        })
        return true
      }
      return false
    },
  })

  // 01 — detail page Permissions card.
  await page.goto(`${base}/apps/detail/${APP}`, { waitUntil: 'domcontentloaded' })
  await page.getByText('Crew Keyboard').first().waitFor({ timeout: 15000 })
  const row = page.locator('code', { hasText: 'sessionApproval' }).first()
  await row.waitFor({ timeout: 15000 })
  await page.waitForTimeout(500)
  // The whole Permissions card: the row must be read next to the api/events lists
  // it extends, so the frame is the card, not the row alone.
  const card = page.locator('div', { has: page.getByText('Permissions', { exact: true }) })
    .filter({ has: row })
    .last()
  assertGrant('detail', await card.innerText(), { manifestKey: true })
  await card.screenshot({ path: `${OUT}/${PREFIX}-01-detail-permissions.png` })

  // 03 — the re-consent notice after a grant-widening update (captured before
  // the modal, which is opened from the same page).
  await page.getByRole('button', { name: /sync|update/i }).first().click()
  const notice = page.getByRole('status').filter({ hasText: /newly asks to control your chats/ })
  await notice.waitFor({ timeout: 15000 })
  const noticeClass = await notice.getAttribute('class')
  if (!noticeClass || !noticeClass.includes('bg-warn-subtle') || noticeClass.includes('bg-ok')) {
    throw new Error(`reconsent notice is not warn-styled: ${noticeClass}`)
  }
  await page.waitForTimeout(400)
  const noticeBox = await notice.boundingBox()
  if (!noticeBox) throw new Error('reconsent notice has no box')
  await page.screenshot({
    path: `${OUT}/${PREFIX}-03-reconsent-notice.png`,
    clip: { x: 0, y: Math.max(0, noticeBox.y - 24), width: 1440, height: Math.min(520, 1100 - Math.max(0, noticeBox.y - 24)) },
  })

  // 02 — consent modal opened from the detail page's own Enable button.
  await page.getByRole('button', { name: /Enable/ }).first().click()
  await page.waitForSelector(MODAL)
  await page.waitForTimeout(400)
  const modalText = await page.locator(MODAL).innerText()
  assertGrant('consent', modalText, { manifestKey: false })
  if (!modalText.includes('This app also asks to control your chats')) {
    throw new Error(`consent: separate grant heading missing: ${modalText}`)
  }
  await page.locator(MODAL).screenshot({ path: `${OUT}/${PREFIX}-02-consent.png` })

  await browser.close()
  srv.close()
  console.log(`Wrote ${OUT}/${PREFIX}-01-detail-permissions.png, ${PREFIX}-02-consent.png and ${PREFIX}-03-reconsent-notice.png`)
}

main().catch(e => { console.error(e); process.exit(1) })
