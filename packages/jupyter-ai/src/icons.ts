// This file is based on iconimports.ts in @jupyterlab/ui-components, but is manually generated.

import { LabIcon } from '@jupyterlab/ui-components';

import chatSvgStr from '../style/icons/chat.svg';
import clouderaCopilotSvg from '../style/icons/cloudera-copilot.svg';
import jupyternautSvg from '../style/icons/jupyternaut.svg';

export const chatIcon = new LabIcon({
  name: 'jupyter-ai::chat',
  svgstr: chatSvgStr
});

export const jupyternautIcon = new LabIcon({
  name: 'jupyter-ai::jupyternaut',
  svgstr: jupyternautSvg
});

export const clouderaCopilotIcon = new LabIcon({
  name: 'jupyter-ai::clouderaCopilot',
  svgstr: clouderaCopilotSvg
});

export const Jupyternaut = jupyternautIcon.react;

export const ClouderaCopilot = clouderaCopilotIcon.react;
