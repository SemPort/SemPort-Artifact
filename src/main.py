import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.prepare import StageError, run_cve_queue
from module.generator.generate import run_cve
from utils.llm import DEFAULT_CONFIG
from utils.logger import get_logger
# from dataset.prepare import main as prepare_main
# from module.locator.anchor import main as anchor_main
# from module.locator.intent import main as intent_main

logger = get_logger(__name__)

_BASE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_BASE)
MODES = ('backport', 'upstream-to-downstream')

def _parse_cves(raw):
    samples = [c.strip() for chunk in raw for c in chunk.split(',') if c.strip()]
    if not samples:
        raise SystemExit('at least one CVE id is required')
    return samples

def _stage_run(args, samples):
    mode = args.mode
    if mode not in MODES:
        raise SystemExit(f'mode must be one of {MODES}')
    # if args.max_retry is not None:
    #     empty_retry = validate_retry = args.max_retry
    # else:
    #     empty_retry = validate_retry = args.validate_retry
    empty_retry = validate_retry = args.validate_retry
    run_name = args.name or 'full'
    started = datetime.datetime.now()
    # output = os.path.abspath(args.output or os.path.join(_REPO_ROOT, 'results', mode))
    output = os.path.abspath(os.path.join(_REPO_ROOT, 'results', mode))

    logger.info(f'CVEs: {len(samples)} | mode={mode} | run={run_name} | model={args.model}')

    def hook(cve, hook_mode):
        try:
            return run_cve(cve, hook_mode or mode, run_name, args.model,
                           args.llm_threads, args.config, args.force,
                           empty_retry=empty_retry, validate_retry=validate_retry,
                           semantic_retry=args.semantic_retry)
        except Exception as exc:
            raise StageError('generate', f'{cve}: {exc}') from exc

    statuses = run_cve_queue(samples, mode, args.threads, run_hook=hook,
                             reprepare=args.force)

    failed = []
    for s in statuses:
        if not s['ok']:
            failed.append(s['cve'])
            logger.info(f'  {s["cve"]:<16s} FAILED at {s["stage"]} ({s["error"]})')
        else:
            logger.info(f'  {s["cve"]:<16s} done')
    payload = {'started_at': started.isoformat(timespec='seconds'),
               'finished_at': datetime.datetime.now().isoformat(timespec='seconds'),
               'mode': mode, 'run': run_name, 'failed': failed,
               'results': statuses}
    os.makedirs(output, exist_ok=True)
    path = os.path.join(output, f'summary_{started.strftime("%Y%m%d-%H%M%S")}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f'failed: {len(failed)} | summary -> {path}')
    return 1 if failed else 0

# def _stage_prepare(args, samples):
#     argv = ['--cve', ','.join(samples), '--mode', args.mode,
#             '--threads', str(args.threads)]
#     if args.force:
#         argv.append('--force')
#     return prepare_main(argv)

# def _stage_intent(args, samples):
#     return intent_main(['--cve', ','.join(samples), '--mode', args.mode,
#                         '--model', args.model, '--llm-threads',
#                         str(args.llm_threads)]
#                        + (['--force'] if args.force else []))

# def _stage_anchor(args, samples):
#     return anchor_main(['--cve', ','.join(samples), '--mode', args.mode,
#                         '--topk', str(args.topk)]
#                        + (['--force'] if args.force else []))

# def _stage_generate(args, samples):
#     run_name = args.name or 'full'
#     for cve in samples:
#         run_cve(cve, args.mode, run_name, args.model, args.llm_threads,
#                 args.config, args.force, empty_retry=args.empty_retry,
#                 validate_retry=args.validate_retry,
#                 semantic_retry=args.semantic_retry)
#     return 0

def main():
    p = argparse.ArgumentParser(prog='main.py')
    sub = p.add_subparsers(dest='cmd', required=True)

    def add_common(sp, with_prepare=False):
        sp.add_argument('cves', nargs='+', help='CVE IDs (space/comma separated)')
        sp.add_argument('--mode', default='backport', choices=MODES)
        sp.add_argument('--force', action='store_true')
        if with_prepare:
            sp.add_argument('--threads', type=int, default=1)

    run = sub.add_parser('run')
    add_common(run, with_prepare=True)
    run.add_argument('--name', default='full')
    run.add_argument('--model', default='GPT-4o-test')
    run.add_argument('--config', default=DEFAULT_CONFIG)
    run.add_argument('--llm-threads', type=int, default=4)
    run.add_argument('--validate-retry', type=int, default=3,
                     help='mechanical-feedback retry budget')
    run.add_argument('--semantic-retry', type=int, default=1,
                     help='semantic-feedback retry budget (unit accounting)')
    # run.add_argument('--max-retry', type=int, default=None)
    # run.add_argument('--output', default=None)

    # prep = sub.add_parser('prepare', help='prepare task data only')
    # add_common(prep, with_prepare=True)

    # intent = sub.add_parser('intent', help='extract repair intent only')
    # add_common(intent)
    # intent.add_argument('--model', default='GPT-4o-test')
    # intent.add_argument('--config', default=DEFAULT_CONFIG)
    # intent.add_argument('--llm-threads', type=int, default=4)

    # anch = sub.add_parser('anchor', help='locate target anchors only')
    # add_common(anch)
    # anch.add_argument('--topk', type=int, default=5)

    # gen = sub.add_parser('generate', help='LLM transform only')
    # add_common(gen)
    # gen.add_argument('--name', default='full')
    # gen.add_argument('--model', default='GPT-4o-test')
    # gen.add_argument('--config', default=DEFAULT_CONFIG)
    # gen.add_argument('--llm-threads', type=int, default=4)
    # gen.add_argument('--empty-retry', type=int, default=1)
    # gen.add_argument('--validate-retry', type=int, default=3,
    #                  help='mechanical-feedback retry budget')
    # gen.add_argument('--semantic-retry', type=int, default=1,
    #                  help='semantic-feedback retry budget (unit accounting)')

    args = p.parse_args()
    samples = _parse_cves(args.cves)
    if args.cmd == 'run':
        return _stage_run(args, samples)
    # if args.cmd == 'prepare':
    #     return _stage_prepare(args, samples)
    # if args.cmd == 'intent':
    #     return _stage_intent(args, samples)
    # if args.cmd == 'anchor':
    #     return _stage_anchor(args, samples)
    # if args.cmd == 'generate':
    #     return _stage_generate(args, samples)

if __name__ == '__main__':
    sys.exit(main())
