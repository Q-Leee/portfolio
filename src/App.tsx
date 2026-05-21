import React, { useState, useEffect, useRef } from 'react';
import { 
  Terminal, 
  Cpu, 
  Layers, 
  Send, 
  Sparkles, 
  MapPin, 
  Mail, 
  Code, 
  Database, 
  ExternalLink,
  Activity
} from 'lucide-react';

// Resume data structures
interface Project {
  id: string;
  title: string;
  subtitle: string;
  role: string;
  period: string;
  location: string;
  description: string;
  bullets: string[];
  tech: string[];
  impact?: string;
  highlighted?: boolean;
}

interface SkillGroup {
  category: string;
  skills: string[];
  icon: React.ReactNode;
}

interface TimelineItem {
  period: string;
  company: string;
  role: string;
  location: string;
  bullets: string[];
  type: 'work' | 'education' | 'military';
}

interface Message {
  id: number;
  sender: 'ai' | 'user';
  text: string;
  timestamp: Date;
}

interface RAGDocChunk {
  title: string;
  score: number;
  content: string;
}

interface AgentLog {
  id: number;
  timestamp: string;
  category: 'AGENT' | 'SYSTEM' | 'EMBEDDING' | 'LOG';
  text: string;
}

// 2D/4D Latent Vector Space Embedding Visualisation structures
interface EmbeddingNode {
  id: string;
  label: string;
  category: 'project' | 'language' | 'database' | 'ai' | 'tool';
  x: number;
  y: number;
  vx: number;
  vy: number;
  baseX: number;
  baseY: number;
  similarity: number;
  details: string;
  targetX?: number;
  targetY?: number;
  radius: number;
}

const vectorData: { [key: string]: number[] } = {
  // AI-related vectors
  'workflow': [0.95, 0.20, 0.80, 0.90],
  'rag': [0.98, 0.10, 0.90, 0.95],
  'vector': [0.92, 0.20, 0.85, 0.88],
  'chromadb': [0.88, 0.30, 0.80, 0.85],
  'ollama': [0.90, 0.20, 0.82, 0.87],
  'pytorch': [0.85, 0.10, 0.70, 0.75],
  'tensorflow': [0.84, 0.10, 0.68, 0.73],
  'prompt': [0.80, 0.40, 0.75, 0.82],

  // Web / Full-Stack / Projects
  'spc': [0.20, 0.95, 0.40, 0.30],
  'fastapi': [0.40, 0.90, 0.60, 0.50],
  'react': [0.10, 0.98, 0.30, 0.20],
  'typescript': [0.30, 0.92, 0.40, 0.30],
  'javascript': [0.20, 0.95, 0.30, 0.20],
  'nodejs': [0.30, 0.88, 0.50, 0.40],
  'turso': [0.20, 0.85, 0.50, 0.30],
  'sheets': [0.10, 0.80, 0.30, 0.10],

  // Mobile / IoT
  'menuscout': [0.20, 0.90, 0.30, 0.40],
  'hope': [0.30, 0.88, 0.40, 0.50],
  'supabase': [0.40, 0.85, 0.60, 0.50],
  'reactnative': [0.20, 0.92, 0.30, 0.40],

  // Enterprise / QA
  'rubicon': [0.10, 0.60, 0.30, 0.20],
  'saperp': [0.10, 0.50, 0.20, 0.10],
  'git': [0.40, 0.70, 0.40, 0.40],
  'docker': [0.50, 0.80, 0.60, 0.50]
};

const getCosineSimilarity = (vecA: number[], vecB: number[]): number => {
  if (!vecA || !vecB) return 0;
  const dotProduct = vecA.reduce((sum, a, idx) => sum + a * vecB[idx], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  if (magnitudeA === 0 || magnitudeB === 0) return 0;
  return parseFloat((dotProduct / (magnitudeA * magnitudeB)).toFixed(4));
};

const mapSkillToNodeId = (skillName: string): string => {
  const norm = skillName.toLowerCase();
  if (norm.includes('fastapi')) return 'fastapi';
  if (norm.includes('react native')) return 'reactnative';
  if (norm.includes('react')) return 'react';
  if (norm.includes('python')) return 'python';
  if (norm.includes('typescript')) return 'typescript';
  if (norm.includes('javascript')) return 'javascript';
  if (norm.includes('supabase')) return 'supabase';
  if (norm.includes('turso')) return 'turso';
  if (norm.includes('sheets')) return 'sheets';
  if (norm.includes('pytorch')) return 'pytorch';
  if (norm.includes('tensorflow')) return 'tensorflow';
  if (norm.includes('rag')) return 'rag';
  if (norm.includes('vector')) return 'vector';
  if (norm.includes('chroma')) return 'chromadb';
  if (norm.includes('ollama')) return 'ollama';
  if (norm.includes('prompt')) return 'prompt';
  if (norm.includes('git')) return 'git';
  if (norm.includes('docker')) return 'docker';
  if (norm.includes('sap')) return 'saperp';
  if (norm.includes('spc') || norm.includes('roster') || norm.includes('assign') || norm.includes('schedule')) return 'spc';
  if (norm.includes('workflow')) return 'workflow';
  if (norm.includes('scout')) return 'menuscout';
  if (norm.includes('hope') || norm.includes('ring')) return 'hope';
  if (norm.includes('rubicon')) return 'rubicon';
  return '';
};

interface VisualizerProps {
  activeFocusId: string | null;
  onNodeClick: (id: string) => void;
}

function EmbeddingSpaceVisualizer({ activeFocusId, onNodeClick }: VisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const hoverStateRef = useRef<EmbeddingNode | null>(null);

  const nodesRef = useRef<EmbeddingNode[]>([]);

  useEffect(() => {
    const rawNodes = [
      { id: 'spc', label: 'SPC Roster App', category: 'project', baseX: 0.15, baseY: 0.2, details: 'Production labour scheduling software proxying Google Sheets.' },
      { id: 'workflow', label: 'WorkFlow AI', category: 'project', baseX: 0.45, baseY: -0.25, details: 'RAG system executing document citations via ChromaDB.' },
      { id: 'menuscout', label: 'Menu Scout', category: 'project', baseX: -0.25, baseY: -0.4, details: 'Expo/React Native macro-nutrition tracker sync via Supabase.' },
      { id: 'hope', label: 'CyterX HOPE', category: 'project', baseX: 0.38, baseY: 0.35, details: 'React Native SOS real-time biometric mapping UI.' },
      { id: 'rubicon', label: 'Rubicon Water', category: 'project', baseX: -0.42, baseY: 0.15, details: 'Embedded irrigation QA software test matrices.' },
      
      { id: 'python', label: 'Python', category: 'language', baseX: 0.1, baseY: -0.15, details: 'Multi-paradigm language for ML pipelines and FastAPI.' },
      { id: 'typescript', label: 'TypeScript', category: 'language', baseX: -0.1, baseY: 0.25, details: 'Typed JavaScript, standard in React SPA & FastAPI hooks.' },
      { id: 'javascript', label: 'JavaScript', category: 'language', baseX: -0.22, baseY: -0.1, details: 'ES6+ standards, powering asynchronous web logic.' },
      { id: 'react', label: 'React', category: 'language', baseX: -0.15, baseY: 0.42, details: 'SPA UI system utilizing hook state matrices.' },
      { id: 'reactnative', label: 'React Native', category: 'language', baseX: 0.18, baseY: -0.35, details: 'Cross-platform native mobile Expo framework builds.' },
      { id: 'nodejs', label: 'Node.js', category: 'language', baseX: -0.05, baseY: -0.3, details: 'V8 server environments syncing REST connections.' },

      { id: 'fastapi', label: 'FastAPI', category: 'database', baseX: 0.22, baseY: -0.05, details: 'Python web API, asynchronous route synchronization.' },
      { id: 'supabase', label: 'Supabase', category: 'database', baseX: 0.25, baseY: -0.5, details: 'BaaS database utilizing Postgres real-time triggers.' },
      { id: 'turso', label: 'Turso', category: 'database', baseX: -0.32, baseY: 0.45, details: 'Edge database using LibSQL replicas and SQLite syncing.' },
      { id: 'sheets', label: 'Google Sheets API', category: 'database', baseX: -0.55, baseY: 0.3, details: 'Enterprise spreadsheet REST synchronization.' },

      { id: 'pytorch', label: 'PyTorch', category: 'ai', baseX: 0.32, baseY: 0.15, details: 'Deep Learning platform, neural matrix calculation.' },
      { id: 'tensorflow', label: 'TensorFlow', category: 'ai', baseX: 0.48, baseY: 0.22, details: 'Graph computational modeling for predictive systems.' },
      { id: 'rag', label: 'RAG Pipelines', category: 'ai', baseX: 0.28, baseY: -0.18, details: 'Retrieval-Augmented Generation context synthesiser.' },
      { id: 'vector', label: 'Vector Search', category: 'ai', baseX: 0.35, baseY: -0.1, details: 'Cosine distance searching across embedding grids.' },
      { id: 'chromadb', label: 'ChromaDB', category: 'ai', baseX: 0.48, baseY: -0.38, details: 'Open-source vector database node matcher.' },
      { id: 'ollama', label: 'Ollama', category: 'ai', baseX: 0.38, baseY: -0.05, details: 'Local inference engine hosting fine-tuned models.' },
      { id: 'prompt', label: 'Prompt Engineering', category: 'ai', baseX: 0.05, baseY: 0.45, details: 'Structural system prompt template orchestration.' },

      { id: 'git', label: 'Git', category: 'tool', baseX: -0.12, baseY: -0.55, details: 'Distributed VCS, merging commits and branch logs.' },
      { id: 'docker', label: 'Docker', category: 'tool', baseX: 0.55, baseY: -0.55, details: 'Container virtualisation compiling server environments.' },
      { id: 'saperp', label: 'SAP ERP', category: 'tool', baseX: -0.62, baseY: -0.15, details: 'Enterprise material logistics and defect tracking logs.' }
    ];

    nodesRef.current = rawNodes.map(node => ({
      ...node,
      category: node.category as 'project' | 'language' | 'database' | 'ai' | 'tool',
      x: (Math.random() - 0.5) * 200,
      y: (Math.random() - 0.5) * 200,
      vx: 0,
      vy: 0,
      targetX: 0,
      targetY: 0,
      similarity: 0,
      radius: node.category === 'project' ? 8 : 6
    }));
  }, []);

  useEffect(() => {
    if (!nodesRef.current.length) return;

    if (activeFocusId) {
      const focusVector = vectorData[activeFocusId];
      nodesRef.current = nodesRef.current.map(node => {
        const nodeVector = vectorData[node.id];
        const similarity = getCosineSimilarity(focusVector, nodeVector);
        return {
          ...node,
          similarity: node.id === activeFocusId ? 1.0 : similarity
        };
      });
    } else {
      nodesRef.current = nodesRef.current.map(node => ({
        ...node,
        similarity: 0
      }));
    }
  }, [activeFocusId]);

  useEffect(() => {
    let animFrameId: number;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let mouseX = -9999;
    let mouseY = -9999;

    const handleMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
        mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
      }
    };

    const handleMouseLeave = () => {
      mouseX = -9999;
      mouseY = -9999;
      hoverStateRef.current = null;
      canvas.style.cursor = 'default';
    };

    const handleCanvasClick = () => {
      const current = hoverStateRef.current;
      if (current) {
        onNodeClick(current.id);
      }
    };

    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseleave', handleMouseLeave);
    canvas.addEventListener('click', handleCanvasClick);

    const bgStars: { x: number; y: number; r: number; alpha: number; speed: number }[] = Array.from({ length: 40 }, () => ({
      x: Math.random() * 600,
      y: Math.random() * 450,
      r: Math.random() * 1.2 + 0.3,
      alpha: Math.random() * 0.7 + 0.2,
      speed: Math.random() * 0.05 + 0.01
    }));

    const updateAndRender = () => {
      const width = canvas.width = canvas.parentElement?.clientWidth || 600;
      const height = canvas.height = 420;
      const centerX = width / 2;
      const centerY = height / 2;

      ctx.fillStyle = '#040814';
      ctx.fillRect(0, 0, width, height);

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.02)';
      ctx.lineWidth = 1;
      const gridSize = 40;
      for (let x = 0; x < width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      if (activeFocusId) {
        ctx.strokeStyle = 'rgba(57, 255, 20, 0.03)';
        ctx.lineWidth = 1;
        [80, 150, 220].forEach((radius, i) => {
          ctx.beginPath();
          ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
          ctx.stroke();
          ctx.fillStyle = 'rgba(57, 255, 20, 0.2)';
          ctx.font = '7px var(--font-mono)';
          ctx.fillText(`SIM < ${[0.90, 0.75, 0.50][i].toFixed(2)}`, centerX + radius - 40, centerY - 4);
        });
      }

      bgStars.forEach(star => {
        star.alpha += star.speed;
        if (star.alpha > 0.95 || star.alpha < 0.1) star.speed = -star.speed;
        ctx.fillStyle = `rgba(255, 255, 255, ${Math.max(0.1, star.alpha)})`;
        ctx.beginPath();
        ctx.arc(star.x % width, star.y % height, star.r, 0, Math.PI * 2);
        ctx.fill();
      });

      let currentHovered: EmbeddingNode | null = null;
      let minDistance = 14;

      const focusNode = nodesRef.current.find(n => n.id === activeFocusId);

      nodesRef.current.forEach(node => {
        let destX, destY;

        if (activeFocusId && focusNode) {
          if (node.id === activeFocusId) {
            destX = centerX;
            destY = centerY;
          } else {
            const simRatio = node.similarity;
            const minRadius = 55;
            const maxRadius = 240;
            const radius = maxRadius - (maxRadius - minRadius) * Math.pow(simRatio, 2.2);

            const angle = Math.atan2(node.baseY, node.baseX);
            destX = centerX + Math.cos(angle) * radius;
            destY = centerY + Math.sin(angle) * radius;
          }
        } else {
          destX = centerX + node.baseX * (width * 0.4);
          destY = centerY + node.baseY * (height * 0.4);
        }

        const stiffness = 0.04;
        const damping = 0.85;

        const forceX = (destX - node.x) * stiffness;
        const forceY = (destY - node.y) * stiffness;

        node.vx = (node.vx + forceX) * damping;
        node.vy = (node.vy + forceY) * damping;

        const time = Date.now() * 0.0006;
        const driftX = Math.sin(time + node.baseX * 100) * 0.08;
        const driftY = Math.cos(time + node.baseY * 100) * 0.08;

        node.x += node.vx + driftX;
        node.y += node.vy + driftY;

        const dx = mouseX - node.x;
        const dy = mouseY - node.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < minDistance) {
          minDistance = dist;
          currentHovered = node;
        }
      });

      if (currentHovered !== hoverStateRef.current) {
        hoverStateRef.current = currentHovered;
        canvas.style.cursor = currentHovered ? 'pointer' : 'default';
      }

      if (activeFocusId && focusNode) {
        nodesRef.current.forEach(node => {
          if (node.id !== activeFocusId && node.similarity > 0.65) {
            const strength = Math.pow((node.similarity - 0.65) / 0.35, 1.5);
            
            ctx.shadowBlur = 10 * strength;
            ctx.shadowColor = 'rgba(57, 255, 20, 0.6)';
            ctx.strokeStyle = `rgba(57, 255, 20, ${0.12 + 0.35 * strength})`;
            ctx.lineWidth = 0.8 + 1.2 * strength;
            
            ctx.beginPath();
            ctx.moveTo(focusNode.x, focusNode.y);
            ctx.lineTo(node.x, node.y);
            ctx.stroke();
            
            ctx.shadowBlur = 0;
          }
        });
      }

      nodesRef.current.forEach(node => {
        const isFocus = node.id === activeFocusId;
        const isHovered = currentHovered && node.id === currentHovered.id;
        
        let nodeColor = 'var(--color-cyan)';
        if (node.category === 'ai') nodeColor = '#a855f7';
        if (node.category === 'language') nodeColor = '#10b981';
        if (node.category === 'database') nodeColor = '#6366f1';
        if (node.category === 'tool') nodeColor = '#3b82f6';

        ctx.beginPath();
        if (isFocus) {
          const scale = 1.0 + Math.sin(Date.now() * 0.007) * 0.15;
          ctx.arc(node.x, node.y, node.radius * 1.5 * scale, 0, Math.PI * 2);
          ctx.fillStyle = nodeColor;
          ctx.shadowBlur = 15;
          ctx.shadowColor = nodeColor;
          ctx.fill();
          ctx.shadowBlur = 0;

          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.radius * 1.5 * scale + 2, 0, Math.PI * 2);
          ctx.stroke();
        } else {
          ctx.arc(node.x, node.y, isHovered ? node.radius * 1.4 : node.radius, 0, Math.PI * 2);
          ctx.fillStyle = nodeColor;
          if (isHovered) {
            ctx.shadowBlur = 10;
            ctx.shadowColor = nodeColor;
          }
          ctx.fill();
          ctx.shadowBlur = 0;
        }

        ctx.fillStyle = isFocus ? '#ffffff' : isHovered ? 'var(--color-cyan)' : 'rgba(255, 255, 255, 0.7)';
        ctx.font = isFocus ? 'bold 10px var(--font-sans)' : '9px var(--font-sans)';
        ctx.textAlign = 'center';
        ctx.fillText(node.label, node.x, node.y - (isFocus ? 16 : 12));
      });

      const activeHovered = currentHovered as EmbeddingNode | null;
      if (activeHovered) {
        ctx.strokeStyle = 'rgba(57, 255, 20, 0.35)';
        ctx.lineWidth = 0.5;
        ctx.setLineDash([2, 3]);

        ctx.beginPath();
        ctx.moveTo(0, activeHovered.y);
        ctx.lineTo(width, activeHovered.y);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(activeHovered.x, 0);
        ctx.lineTo(activeHovered.x, height);
        ctx.stroke();
        
        ctx.setLineDash([]);

        const hudX = Math.min(width - 190, Math.max(10, activeHovered.x - 90));
        const hudY = activeHovered.y > height - 100 ? activeHovered.y - 85 : activeHovered.y + 15;

        ctx.fillStyle = 'rgba(3, 7, 18, 0.9)';
        ctx.strokeStyle = 'var(--color-cyan)';
        ctx.lineWidth = 1;
        
        ctx.beginPath();
        ctx.roundRect(hudX, hudY, 180, 68, 6);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = 'var(--color-cyan)';
        ctx.font = 'bold 8px var(--font-mono)';
        ctx.textAlign = 'left';
        ctx.fillText(`CONCEPT: ${activeHovered.label.toUpperCase()}`, hudX + 10, hudY + 16);

        ctx.fillStyle = '#ffffff';
        ctx.font = '7px var(--font-mono)';
        ctx.fillText(`VECTOR: [${vectorData[activeHovered.id] ? vectorData[activeHovered.id].join(', ') : '0.0, 0.0'}]`, hudX + 10, hudY + 28);

        ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
        ctx.font = '6.5px var(--font-sans)';
        
        const words = activeHovered.details.split(' ');
        let line = '';
        let lineCount = 0;
        for (let n = 0; n < words.length; n++) {
          let testLine = line + words[n] + ' ';
          let metrics = ctx.measureText(testLine);
          if (metrics.width > 160 && n > 0) {
            ctx.fillText(line, hudX + 10, hudY + 40 + lineCount * 10);
            line = words[n] + ' ';
            lineCount++;
          } else {
            line = testLine;
          }
        }
        ctx.fillText(line, hudX + 10, hudY + 40 + lineCount * 10);
      }

      animFrameId = requestAnimationFrame(updateAndRender);
    };

    updateAndRender();

    return () => {
      cancelAnimationFrame(animFrameId);
      canvas.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('mouseleave', handleMouseLeave);
      canvas.removeEventListener('click', handleCanvasClick);
    };
  }, [activeFocusId, onNodeClick]);

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'relative' }}>
      <canvas ref={canvasRef} style={{ display: 'block', width: '100%', height: '100%' }} />
    </div>
  );
}


export default function App() {
  const [selectedSkill, setSelectedSkill] = useState<string | null>(null);
  const [chatInput, setChatInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      sender: 'ai',
      text: `Hello! 👋 I am Q-AI, the virtual assistant for HyungKyu (Q) Lee, an Agile-oriented Full Stack & AI Engineer based in Shepparton, VIC, Australia.

I am fully loaded with Q's professional resume. I can instantly answer your questions about his production projects, core skills, Australian visa work rights, or contact information!

Click one of the suggested topics below or type your question directly in the chat!`,
      timestamp: new Date()
    }
  ]);
  const [isTyping, setIsTyping] = useState(false);
  const [copiedEmail, setCopiedEmail] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [modalContent, setModalContent] = useState({ title: '', message: '', subtext: '' });

  // RAG Pipeline states
  const [activeTab, setActiveTab] = useState<'chat' | 'rag'>('chat');
  const [ragStep, setRagStep] = useState<number>(0); // 0: Idle, 1: Tokenizing, 2: Embedding, 3: Retrieving, 4: Reranking, 5: Synthesizing
  const [activeQuery, setActiveQuery] = useState('');
  const [processedTokens, setProcessedTokens] = useState<string[]>([]);
  const [embeddingVector, setEmbeddingVector] = useState<number[]>([]);
  const [retrievedDocs, setRetrievedDocs] = useState<RAGDocChunk[]>([]);
  const [synthesisPrompt, setSynthesisPrompt] = useState('');
  const [activeEmbeddingFocus, setActiveEmbeddingFocus] = useState<string | null>(null);

  // Agent Telemetry HUD states & refs
  const [hudCollapsed, setHudCollapsed] = useState(false);
  
  // Contact Form states
  const [contactName, setContactName] = useState('');
  const [contactEmail, setContactEmail] = useState('');
  const [contactSubject, setContactSubject] = useState('');
  const [contactMessage, setContactMessage] = useState('');
  const [contactSending, setContactSending] = useState(false);

  const [agentLogs, setAgentLogs] = useState<AgentLog[]>([
    { id: 1, timestamp: new Date().toLocaleTimeString(), category: 'SYSTEM', text: 'Initializing Q-Agent-Core v2.4...' },
    { id: 2, timestamp: new Date().toLocaleTimeString(), category: 'EMBEDDING', text: 'Pre-loading sparse BM25 candidate dictionaries...' },
    { id: 3, timestamp: new Date().toLocaleTimeString(), category: 'AGENT', text: 'Ready. Monitoring recruiter engagement metrics.' }
  ]);

  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const logsContainerRef = useRef<HTMLDivElement>(null);

  // Helper to add agent log
  const addAgentLog = (category: 'AGENT' | 'SYSTEM' | 'EMBEDDING' | 'LOG', text: string) => {
    setAgentLogs(prev => {
      const newLog = {
        id: prev.length + 1,
        timestamp: new Date().toLocaleTimeString(),
        category,
        text
      };
      return [...prev, newLog].slice(-25); // Keep last 25 logs
    });
  };

  // Scroll to bottom of agent logs
  useEffect(() => {
    if (logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
    }
  }, [agentLogs, hudCollapsed]);

  // Auto scroll chat
  useEffect(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTo({
        top: messagesContainerRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [messages, isTyping]);

  // Monitor recruiter scroll behavior to trigger live telemetry
  useEffect(() => {
    let lastSection = '';
    const handleScroll = () => {
      const sections = ['about', 'projects', 'skills', 'timeline', 'contact'];
      let currentSection = '';
      
      for (const section of sections) {
        const el = document.getElementById(section);
        if (el) {
          const rect = el.getBoundingClientRect();
          if (rect.top <= window.innerHeight * 0.45 && rect.bottom >= window.innerHeight * 0.45) {
            currentSection = section;
            break;
          }
        }
      }
      
      if (currentSection && currentSection !== lastSection) {
        lastSection = currentSection;
        const capitalized = currentSection.charAt(0).toUpperCase() + currentSection.slice(1);
        addAgentLog('AGENT', `User scrolled to ${capitalized} Section.`);
        if (currentSection === 'projects') {
          addAgentLog('SYSTEM', 'Vector similarity search active (Focus: Full-Stack automation).');
          addAgentLog('LOG', "Detected high interest in 'SPC Job Assignment App' (Score: 0.98).");
        } else if (currentSection === 'skills') {
          addAgentLog('EMBEDDING', 'Pre-calculating distance metrics for skill badges...');
        } else if (currentSection === 'contact') {
          addAgentLog('AGENT', 'Recommendation: Suggesting sponsorship visa query to user.');
        }
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Skill matrix highlight link logic
  const handleSkillClick = (skill: string) => {
    if (selectedSkill === skill) {
      setSelectedSkill(null); // Deselect if clicked again
      setActiveEmbeddingFocus(null);
      addAgentLog('SYSTEM', 'Cleared skills filter matrix.');
    } else {
      setSelectedSkill(skill);
      addAgentLog('SYSTEM', `Filter node activated: '${skill}'.`);
      addAgentLog('EMBEDDING', `Calculating cosine distances for cluster containing "${skill}"...`);
      addAgentLog('LOG', `Highlighting project assets linked to '${skill}'. Match score: 0.99`);
      
      const nodeId = mapSkillToNodeId(skill);
      if (nodeId) {
        setActiveEmbeddingFocus(nodeId);
      }

      // Smooth scroll to projects section
      const projectsSec = document.getElementById('projects');
      if (projectsSec) {
        projectsSec.scrollIntoView({ behavior: 'smooth' });
      }
    }
  };

  // Copy Email to clipboard utility
  const handleCopyEmail = () => {
    navigator.clipboard.writeText('hyungkyu.lee.q@gmail.com');
    setCopiedEmail(true);
    addAgentLog('LOG', 'Security event: copied contact email hyungkyu.lee.q@gmail.com.');
    addAgentLog('AGENT', 'Generating secure handshake token. SMTP socket prepared.');
    setTimeout(() => setCopiedEmail(false), 2000);
  };

  // Q-AI chatbot response matchmaking engine with real-time Visual RAG Pipeline simulation
  const triggerAIResponse = (userText: string) => {
    setIsTyping(true);
    setActiveQuery(userText);
    setRagStep(1); // 1. Tokenizing
    
    // Auto shift active embedding focus if query matches key concepts
    const matchedNode = mapSkillToNodeId(userText);
    if (matchedNode) {
      setActiveEmbeddingFocus(matchedNode);
      addAgentLog('EMBEDDING', `Auto-shifting Latent Vector focus to: [${matchedNode}] based on query match.`);
    }

    // Auto switch tab to RAG on mobile so user sees the cool telemetry
    if (window.innerWidth <= 768) {
      setActiveTab('rag');
    }

    // Step 1: Tokenization
    const tokens = userText
      .replace(/[.,/#!$%^&*;:{}=\-_`~()?]/g, "")
      .split(/\s+/)
      .filter(t => t.length > 0);
    setProcessedTokens(tokens);
    addAgentLog('SYSTEM', `Tokenizing search query parameters: [${tokens.join(', ')}]`);
    
    // Step 2: Embedding Generation (starts at 500ms)
    setTimeout(() => {
      setRagStep(2);
      // Generate a mock 8-dimensional float vector based on the query string
      const vector = Array.from({ length: 8 }, (_, i) => {
        const hash = userText.split('').reduce((acc, char) => acc + char.charCodeAt(0) * (i + 1), 0);
        const floatVal = ((hash % 200) - 100) / 100;
        return parseFloat(floatVal.toFixed(4));
      });
      setEmbeddingVector(vector);
      addAgentLog('EMBEDDING', 'Recalculating distance metrics: projected query into 8D vector space.');
    }, 500);

    // Step 3: Vector & Keyword Retrieval (starts at 1000ms)
    setTimeout(() => {
      setRagStep(3);
      const query = userText.toLowerCase().trim();
      let mockDocs: RAGDocChunk[] = [];
      
      if (query.includes("spc") || query.includes("roster") || query.includes("assign") || query.includes("schedule") || query.includes("board")) {
        mockDocs = [
          { title: "SPC Job Assignment App Specs", score: 0.942, content: "Developed a live production roster and assignment board using React, FastAPI, and Turso to replace spreadsheet overhead." },
          { title: "Torrens University AI Degree", score: 0.581, content: "Coursework covering database engineering, software lifecycle, and enterprise integrations." },
          { title: "Rubicon Water embedded QA logs", score: 0.412, content: "Inventory tracking and Defect logging using SAP ERP systems." }
        ];
      } else if (query.includes("workflow") || query.includes("rag") || query.includes("ollama") || query.includes("search") || query.includes("vector") || query.includes("ai") || query.includes("ml")) {
        mockDocs = [
          { title: "WorkFlow AI - RAG Matcher Architecture", score: 0.968, content: "Hybrid search pipeline merging dense embeddings with BM25 keyword matching and Cross-Encoder reranking." },
          { title: "Torrens University AI Degree", score: 0.745, content: "Extensive Machine Learning, Natural Language Processing, and computer vision coursework." },
          { title: "CyterX HOPE Biometric distress models", score: 0.512, content: "Biometric distress signals and emergency IoT telemetry logic mapped with Supabase." }
        ];
      } else if (query.includes("sponsorship") || query.includes("visa") || query.includes("work rights") || query.includes("australia") || query.includes("availability")) {
        mockDocs = [
          { title: "Q's Australian Visa & Work Rights", score: 0.954, content: "Subclass 485 Temporary Graduate Visa. Full working rights without hourly limitations. Ready for immediate relocation." },
          { title: "ACS Professional Year Program VIC", score: 0.612, content: "Accredited training program targeting Australian corporate standards, business communication, and Scrum methodologies." },
          { title: "Premier Fresh Team Leader logistics", score: 0.385, content: "Logistical resource management and shift allocations across Melbourne sites." }
        ];
      } else if (query.includes("skill") || query.includes("tech") || query.includes("languages") || query.includes("framework")) {
        mockDocs = [
          { title: "Q's core technical profile", score: 0.923, content: "Proficient in Python, TS/JS, React, React Native, FastAPI, ChromaDB, vector search, and local Ollama pipelines." },
          { title: "SPC Roster App Stack", score: 0.784, content: "React, TypeScript, FastAPI, Turso DB, SQLite, Google Sheets REST API, and Netlify hosting." },
          { title: "CyterX HOPE IoT Ecosystem", score: 0.615, content: "Supabase authentication, real-time channels, and React Native mobile builds." }
        ];
      } else if (query.includes("menu") || query.includes("scout") || query.includes("hope") || query.includes("smart ring") || query.includes("rubicon") || query.includes("testing")) {
        mockDocs = [
          { title: "CyterX HOPE Smart Ring Specs", score: 0.912, content: "Co-developed biometric distress mapping dashboard and React Native application with Supabase integration." },
          { title: "Menu Scout Nutrition App", score: 0.884, content: "Cross-platform mobile application utilizing Expo and Supabase for food macros and pricing comparison." },
          { title: "Rubicon Water QA systems", score: 0.852, content: "Embedded software tester. Executed hardware test logs and logged regression discrepancies." }
        ];
      } else if (query.includes("contact") || query.includes("email") || query.includes("phone") || query.includes("linkedin")) {
        mockDocs = [
          { title: "Q's Direct Contact Information", score: 0.985, content: "Email: hyungkyu.lee.q@gmail.com. LinkedIn: linkedin.com/in/qleeq. GitHub: github.com/Q-Leee. Mobile options available." },
          { title: "ACS Professional Year VIC", score: 0.452, content: "Focus on corporate communication, local business networking, and professional ethics." }
        ];
      } else if (query.includes("army") || query.includes("military") || query.includes("squad leader")) {
        mockDocs = [
          { title: "ROK Army Squad Leader Service", score: 0.975, content: "Managed team of 10 squad members in high-pressure tactical environments, building extreme discipline and planning skills." }
        ];
      } else {
        mockDocs = [
          { title: "Q's General Portfolio Profile", score: 0.850, content: "AI & Full-Stack developer based in Australia, focused on RAG engines, database sync, and industrial automation." },
          { title: "WorkFlow AI Project", score: 0.720, content: "FastAPI, Ollama local inference, semantic candidate screening, and vector embeddings." }
        ];
      }
      setRetrievedDocs(mockDocs);
      addAgentLog('SYSTEM', `Vector & Keyword search returned ${mockDocs.length} persistent ChromaDB nodes.`);
    }, 1000);

    // Step 4: Cross-Encoder Reranking (starts at 1500ms)
    setTimeout(() => {
      setRagStep(4);
      setRetrievedDocs(prev => {
        const adjusted = prev.map(d => ({
          ...d,
          score: parseFloat(Math.min(0.999, d.score + (d.title.toLowerCase().includes("specs") || d.title.toLowerCase().includes("rights") || d.title.toLowerCase().includes("profile") || d.title.toLowerCase().includes("contact") || d.title.toLowerCase().includes("service") ? 0.015 : -0.02)).toFixed(3))
        }));
        return [...adjusted].sort((a, b) => b.score - a.score);
      });
      addAgentLog('SYSTEM', 'Cross-Encoder Reranker active: re-weighting semantic similarity scoring.');
    }, 1500);

    // Step 5: Prompt Synthesis & Model Input (starts at 2000ms)
    setTimeout(() => {
      setRagStep(5);
      const query = userText.toLowerCase().trim();
      let response = '';

      if (query.includes("spc") || query.includes("roster") || query.includes("assign") || query.includes("schedule") || query.includes("board")) {
        response = `🤖 [SPC Job Assignment Web App]
Q designed and maintains a live production roster & job assignment web application for SPC warehouse and production teams.

• Tech Stack: TypeScript, React, Vite, Python (FastAPI), Google Sheets API, Turso (LibSQL), Netlify
• Key Achievements:
  - Replaced manual spreadsheet operations with a dynamic drag-and-drop assignment board, saving planning time and improving operational accuracy.
  - Developed a robust Python (FastAPI) proxy service using Google Sheets API as the live data layer to align with existing spreadsheet workflows.
  - Implemented Turso (LibSQL) for persistent logging of roster history, configurations, and user session data.
  - Deployed on Netlify with automated environment-based builds for on-site supervisors.`;
      } 
      else if (query.includes("workflow") || query.includes("rag") || query.includes("ollama") || query.includes("search") || query.includes("vector") || query.includes("ai") || query.includes("ml")) {
        response = `🧠 [WorkFlow AI & Machine Learning]
A high-performance, full-stack RAG (Retrieval-Augmented Generation) application designed by Q for document analysis and CV matchmaking.

• Tech Stack: FastAPI (Python), React, TypeScript, Vite, Chroma DB, Ollama, BM25 Cross-Encoder
• Core Strengths:
  - Designed a hybrid search pipeline combining dense vector embeddings with sparse BM25 (Reciprocal Rank Fusion) and Cross-Encoder reranking for pinpoint retrieval.
  - Crafted prompt workflows utilizing fact extraction and two-stage cited answering to eliminate LLM hallucinations.
  - Engineered resume–JD alignment: parses posting requirements, scores semantic fit, and generates explainable strength/gap summaries for screening.`;
      }
      else if (query.includes("sponsorship") || query.includes("visa") || query.includes("work rights") || query.includes("australia") || query.includes("availability")) {
        response = `🇦🇺 [Visa Status & Work Rights]
Q is legally authorized to work in Australia and is ready for immediate onboarding.

• Visa Status: Subclass 485 Temporary Graduate Visa.
• Work Rights: Full working rights in Australia (no hour restrictions).
• Availability: Immediate start.
• Location: Shepparton, VIC (open to relocation across Australia and 100% remote positions).
• Sponsorship: Actively seeking opportunities linked to technical visa sponsorship (e.g., TSS 482 / PR pathways) for long-term career growth.`;
      }
      else if (query.includes("skill") || query.includes("tech") || query.includes("languages") || query.includes("framework")) {
        response = `💻 [Core Technical Stack]
• Programming: Python, TypeScript, JavaScript (ES6+), React, React Native, Node.js, Express, HTML5, CSS3
• Databases & Cloud: Supabase, Turso (LibSQL), MySQL, MongoDB, Google Sheets API, REST APIs
• AI & Machine Learning: PyTorch, TensorFlow, Pandas, NumPy, OpenCV, RAG Pipelines, Vector Search, ChromaDB, Ollama Local Inference
• Tools & Methodologies: Git, GitHub, VS Code, Docker, GitHub Actions, Jest, Agile/Scrum, SAP ERP`;
      }
      else if (query.includes("menu") || query.includes("scout") || query.includes("hope") || query.includes("smart ring") || query.includes("rubicon") || query.includes("testing")) {
        response = `📱 [Other Major Projects & QA Testing]
• Menu Scout: Cross-platform mobile app built with React Native, Expo, and Supabase to compare store menu items by macro-nutrition and price.
• CyterX HOPE Smart Ring: Co-developed the IoT web dashboard, marketing landing page, and React Native mobile app utilizing Supabase for biometric tracking and SOS alert workflows.
• Rubicon Water: Embedded systems software tester who performed regression testing on irrigation controllers, documented defects, and managed warehouse inventory via SAP ERP.`;
      }
      else if (query.includes("contact") || query.includes("email") || query.includes("phone") || query.includes("linkedin")) {
        response = `📞 [Contact Details]
Q welcomes interview inquiries, technical discussions, and coffee chats!

• Email: hyungkyu.lee.q@gmail.com
• LinkedIn: linkedin.com/in/qleeq
• GitHub: github.com/Q-Leee
• Current Location: Shepparton, Victoria, Australia
• Availability: Immediate Full-Time`;
      }
      else if (query.includes("army") || query.includes("military") || query.includes("squad leader")) {
        response = `🪖 [ROK Army Squad Leader]
Q served as a Squad Leader in the Republic of Korea Army (2013-2015). Under high-pressure environments, he managed squad operations, coordinated tactical team maneuvers, and honed vital leadership, discipline, and accountability skills.`;
      }
      else {
        response = `👋 Hello! I am here to help you learn more about Q's engineering experience.

Try selecting one of these popular questions:
• "Tell me about the SPC Roster Web App"
• "Explain the WorkFlow AI (RAG) Project"
• "What is Q's visa & sponsorship status in Australia?"
• "Show me Q's technical skill set"`;
      }

      const promptTemplate = `[SYSTEM_PROMPT]
      Act as Q-AI, the virtual assistant for HyungKyu (Q) Lee. Synthesize a professional response.
      
      [RETRIEVED_CONTEXT]
      ${retrievedDocs.map((d, index) => `Doc ${index+1}: [${d.title}] ${d.content}`).join("\n")}
      
      [USER_QUESTION]
      "${userText}"
      
      [INFERENCE_EXECUTION]
      Ollama Local / Llama-3-Q-FineTune (8B) temperature=0.2...`;
 
       setSynthesisPrompt(promptTemplate);
       addAgentLog('AGENT', 'Context synthesised. Submitting citation prompt to Ollama Llama-3 model.');
 
       // Complete the pipeline and stream into messages
       setTimeout(() => {
         setMessages(prev => [
           ...prev,
           {
             id: prev.length + 1,
             sender: 'ai',
             text: response,
             timestamp: new Date()
           }
         ]);
         setIsTyping(false);
         setRagStep(0);
         addAgentLog('LOG', `Answer streams complete. Cosine similarity threshold validated.`);
         
         // Auto return to chat tab on mobile so user sees the response
         if (window.innerWidth <= 768) {
           setActiveTab('chat');
         }
       }, 500);
 
     }, 2000);
  };

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    const userText = chatInput;
    setMessages(prev => [
      ...prev,
      {
        id: prev.length + 1,
        sender: 'user',
        text: userText,
        timestamp: new Date()
      }
    ]);
    setChatInput('');
    addAgentLog('LOG', `Recruiter query submitted: "${userText}"`);
    triggerAIResponse(userText);
  };

  const handleSuggestionClick = (question: string) => {
    setMessages(prev => [
      ...prev,
      {
        id: prev.length + 1,
        sender: 'user',
        text: question,
        timestamp: new Date()
      }
    ]);
    addAgentLog('LOG', `Recruiter clicked suggestion chip: "${question}"`);
    triggerAIResponse(question);
  };

  // Projects list
  const projects: Project[] = [
    {
      id: 'spc',
      title: 'SPC Job Assignment Web App',
      subtitle: 'Production App for Daily Operations & Planning',
      role: 'Automation / Software Engineer',
      period: 'Mar 2026 – Present',
      location: 'Shepparton, VIC',
      description: 'Designed and built a live production roster and assignment board web application to manage daily labour allocation and shift schedules for SPC warehouse and manufacturing teams.',
      bullets: [
        'Migrated the system architecture from Python/Streamlit to React SPA + FastAPI for enhanced separation of concerns, scalability, and seamless supervisor interfaces.',
        'Integrated Google Sheets API as the operational cache layer, ensuring real-time alignment with existing spreadsheets and eliminating redundant manual entries.',
        'Engineered a Turso (LibSQL) persistence layer to log historical roster changes, supervisor session metadata, and shift configurations.',
        'Developed an interactive drag-and-drop shift board UI and staffing analytics charts (reusable layout components).',
        'Implemented Google OAuth browser-based editing alongside a read-only proxy path specifically optimized for warehouse wall display systems.',
        'Adopted an Agile workflow, delivering rapid updates and enhancements based on direct floor feedback.'
      ],
      tech: ['React', 'TypeScript', 'Vite', 'FastAPI', 'Python', 'Google Sheets API', 'Turso', 'LibSQL', 'Netlify', 'SQL', 'Agile/Scrum'],
      impact: '🔥 Saved countless hours of manual spreadsheet updates and improved daily scheduling reliability for SPC warehouse and production planning.',
      highlighted: true
    },
    {
      id: 'workflow',
      title: 'WorkFlow AI',
      subtitle: 'Advanced PDF Q&A & Resume-JD Semantic Matcher',
      role: 'Personal Project',
      period: '2025 – Present',
      location: 'Remote',
      description: 'A full-stack RAG (Retrieval-Augmented Generation) application designed to facilitate document Q&A, automatic meeting summaries, and semantic candidate-to-posting evaluation.',
      bullets: [
        'Architected a FastAPI backend and React frontend that orchestrates local LLM inference via Ollama.',
        'Engineered a hybrid retrieval engine merging dense semantic embeddings with sparse BM25 search (Reciprocal Rank Fusion) and Cross-Encoder reranking.',
        'Designed custom prompt flows utilizing factual extraction and two-stage response generation to reduce LLM hallucinations.',
        'Developed resume-JD alignment: parses postings, handles per-requirement semantic matching, and outputs gap/strength analytics.',
        'Integrated Chroma DB for vector storage, JWT auth security, and customizable document chunking.'
      ],
      tech: ['FastAPI', 'Python', 'React', 'TypeScript', 'Vite', 'Chroma DB', 'Ollama', 'RAG Pipelines', 'Vector Search', 'NLP', 'Prompt Engineering'],
      impact: '🧠 Achieved highly accurate retrieval performance and cited source responses on local hardware without expensive cloud model APIs.',
      highlighted: true
    },
    {
      id: 'menuscout',
      title: 'Menu Scout',
      subtitle: 'Cross-platform Mobile Nutrition & Price Tracker',
      role: 'Personal Project',
      period: 'Nov 2025 – Present',
      location: 'Melbourne, VIC',
      description: 'A cross-platform mobile application designed to help health-conscious users compare local menu items by macro-nutrients and price points.',
      bullets: [
        'Designed and compiled a cross-platform mobile app using React Native and Expo.',
        'Integrated a Supabase real-time backend to power secure user authentication, favorited items, and cross-device sync.',
        'Created high-performance UI layouts and streamlined stack navigation to ensure frictionless user journeys.'
      ],
      tech: ['React Native', 'Expo', 'Supabase', 'TypeScript', 'JavaScript (ES6+)'],
      impact: '📱 Streamlined mobile navigation flows and optimized responsive component structures for a premium user experience.'
    },
    {
      id: 'hope',
      title: 'CyterX HOPE Smart Ring',
      subtitle: 'Biometric Security & Distress SOS Web/App Ecosystem',
      role: 'Junior Full Stack Developer',
      period: 'Jul 2025 – Nov 2025',
      location: 'Melbourne, VIC',
      description: 'Contributed to the development of the HOPE smart ring ecosystem across its marketing landing, central user dashboard, and companion mobile application.',
      bullets: [
        'Built responsive web interfaces illustrating AI distress detection systems and emergency SOS workflows.',
        'Designed dashboard components that represent live biometric telemetry and real-time location coordinate mapping.',
        'Implemented secure database tables and data flows with Supabase to manage IoT user metrics.',
        'Collaborated closely with founders and UX designers in a lean team to rapidly test and iterate concepts.'
      ],
      tech: ['React', 'React Native', 'Node.js', 'Supabase', 'JavaScript (ES6+)'],
      impact: '💍 Successfully delivered MVP-ready screens and biometric tracking components ahead of the public product launch.'
    },
    {
      id: 'rubicon',
      title: 'Rubicon Water Embedded Tester',
      subtitle: 'Embedded Irrigation Controller QA & ERP Management',
      role: 'Software Tester (Embedded Systems)',
      period: 'Mar 2025 – Oct 2025',
      location: 'Shepparton, VIC',
      description: 'Executed functional and regression tests on automated embedded irrigation controllers to guarantee flawless operations under tough environmental conditions.',
      bullets: [
        'Formulated rigorous test plans and executed testing cycles on embedded hardware controllers.',
        'Logged defects and regression results in detail, minimizing debugging cycles for hardware engineers.',
        'Managed inventory movements, component tracking, and material logistics processes utilizing SAP ERP.'
      ],
      tech: ['SAP ERP', 'Embedded Systems Testing', 'QA Processes', 'Agile/Scrum'],
      impact: '⚙️ Identified critical firmware discrepancies prior to deployment, ensuring stable agricultural field performance.'
    }
  ];

  // Skill matrix groups
  const skillGroups: SkillGroup[] = [
    {
      category: 'Programming & Frameworks',
      skills: ['Python', 'TypeScript', 'JavaScript (ES6+)', 'React', 'React Native', 'Node.js', 'Express.js', 'HTML5', 'CSS3', 'Vite'],
      icon: <Code style={{ color: 'var(--color-cyan)' }} size={20} />
    },
    {
      category: 'Backend, Data & Cloud',
      skills: ['FastAPI', 'REST APIs', 'Supabase', 'Turso', 'LibSQL', 'Google Sheets API', 'MySQL', 'MongoDB', 'SQL', 'Netlify', 'Vercel'],
      icon: <Database style={{ color: 'var(--color-indigo)' }} size={20} />
    },
    {
      category: 'AI / Machine Learning',
      skills: ['PyTorch', 'TensorFlow', 'NumPy', 'Pandas', 'scikit-learn', 'OpenCV', 'NLP', 'RAG Pipelines', 'Vector Search', 'Ollama', 'Prompt Engineering'],
      icon: <Cpu style={{ color: 'var(--color-emerald)' }} size={20} />
    },
    {
      category: 'Tools & Enterprise',
      skills: ['Git', 'GitHub', 'Docker', 'VS Code', 'GitHub Actions', 'SAP ERP', 'Warehouse Operations', 'Logistics Workflows', 'Power Automate', 'Agile/Scrum'],
      icon: <Layers style={{ color: 'var(--color-purple)' }} size={20} />
    }
  ];

  // Timeline data
  const timeline: TimelineItem[] = [
    {
      period: 'Mar 2026 – Present',
      company: 'SPC Group',
      role: 'Automation / Software Engineer',
      location: 'Shepparton, VIC',
      bullets: [
        'Engineered a production-level React + FastAPI job assignment web app (utilizing Vite, Google Sheets API, and Turso DB) to optimize shift allocations.',
        'Maintained agile feedback cycles with floor supervisors to drive continuous, weekly pipeline updates and feature adjustments.'
      ],
      type: 'work'
    },
    {
      period: 'Feb 2026 – Mar 2026',
      company: 'SPC Group',
      role: 'Forklift Driver – Warehouse and Logistics',
      location: 'Shepparton, VIC',
      bullets: [
        'Supported high-volume logistics and warehouse operations in a major food manufacturing plant, coordinating pallet transit, staging, and cargo loading.',
        'Acquired deep operational domain knowledge, directly motivating the creation of custom automation software to eliminate manual paperwork.'
      ],
      type: 'work'
    },
    {
      period: 'Mar 2025 – Oct 2025',
      company: 'Rubicon Water',
      role: 'Software Tester (Embedded Systems)',
      location: 'Shepparton, VIC',
      bullets: [
        'Validated embedded irrigation controller reliability through systematic QA test execution.',
        'Utilized SAP ERP to keep track of high-value inventory parts and maintain operational records.'
      ],
      type: 'work'
    },
    {
      period: 'Jul 2025 – Nov 2025',
      company: 'CyterX HOPE',
      role: 'Junior Full Stack Developer',
      location: 'Melbourne, VIC',
      bullets: [
        'Developed real-time biometric and location-tracking dashboard modules linked to emergency SOS services.',
        'Assisted start-up stakeholders to iterate and deliver MVP features through rapid prototype development.'
      ],
      type: 'work'
    },
    {
      period: 'Aug 2017 – Jul 2025',
      company: 'Premier Fresh',
      role: 'Team Leader – Logistics Operations',
      location: 'Epping, VIC',
      bullets: [
        'Led day-to-day warehouse operations, oversaw shipping allocations, and optimized schedules using WMS data.',
        'Created custom spreadsheet formulas and macros to automate internal performance reports, saving weekly planning hours.'
      ],
      type: 'work'
    },
    {
      period: 'Mar 2025 – Present',
      company: 'ACS (Australian Computer Society)',
      role: 'Professional Year Program',
      location: 'Melbourne, VIC',
      bullets: [
        'Participating in accredited professional development targeting Australian corporate culture, business communication, and industry standards.'
      ],
      type: 'education'
    },
    {
      period: 'Nov 2021 – Oct 2024',
      company: 'Torrens University',
      role: 'Bachelor of Software Engineering (Artificial Intelligence)',
      location: 'Melbourne, VIC',
      bullets: [
        'Completed comprehensive coursework covering Machine Learning, Computer Vision, Natural Language Processing (NLP), Database Systems, and C Programming.'
      ],
      type: 'education'
    },
    {
      period: '2013 – 2015',
      company: 'Republic of Korea Army',
      role: 'Squad Leader',
      location: 'South Korea',
      bullets: [
        'Led a military squad in high-stakes environments, cultivating robust discipline, strategic coordination, and strong team responsibility.'
      ],
      type: 'military'
    }
  ];

  return (
    <div className="app-container">
      {/* Dynamic Glow Accents */}
      <div className="glow-accent glow-cyan"></div>
      <div className="glow-accent glow-indigo"></div>

      {/* Floating Header */}
      <header className="nav-header">
        <div className="max-width-container">
          <div className="nav-glass glass">
            <a href="#" className="nav-logo">
              <Terminal size={22} className="text-cyan-400 animate-pulse" />
              <span>Q.LEE</span>
            </a>
            
            <nav className="nav-links">
              <a href="#about">About</a>
              <a href="#projects">Projects</a>
              <a href="#skills">Skills</a>
              <a href="#timeline">Timeline</a>
              <a href="#contact">Contact</a>
            </nav>

            <div className="nav-actions">
              <div className="sponsorship-badge">
                <span className="sponsorship-dot"></span>
                <span>Sponsorship Ready</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section id="about" className="hero-section">
        <div className="max-width-container max-width-wide">
          <div className="hero-grid">
            
            {/* Left Column: Hero Text */}
            <div className="hero-content">
              <div className="hero-pretitle">Agile Full Stack / AI Engineer</div>
              <h1 className="hero-title">
                Engineering <br />
                <span className="highlight-text">Practical Solutions</span><br />
                with Full Stack & AI.
              </h1>
              <p className="hero-subtitle">
                I combine robust software engineering with real-world logistics and operational experience. From launching live production systems (SPC Roster App) to building high-performance local RAG engines (WorkFlow AI), I deliver practical, high-impact applications.
              </p>

              <div className="hero-badges-row">
                <span className="badge badge-cyan">Full Working Rights (485)</span>
                <span className="badge badge-indigo">Immediate Start</span>
                <span className="badge badge-emerald">Melbourne / Relocation</span>
                <span className="badge badge-purple">Sponsorship Sought</span>
              </div>

              <div className="hero-buttons">
                <a href="#projects" className="btn-primary">
                  <Sparkles size={18} />
                  <span>Explore Projects</span>
                </a>
                <a href="#contact" className="btn-secondary">
                  <Mail size={18} />
                  <span>Get in Touch</span>
                </a>
              </div>
            </div>

            {/* Right Column: Q-AI Interactive Chatbot & RAG Dashboard Pane */}
            <div className="ai-dashboard-container glass">
              
              {/* Tab Switcher for Mobile Responsiveness */}
              <div className="dashboard-tabs">
                <button 
                  className={`dashboard-tab ${activeTab === 'chat' ? 'active' : ''}`}
                  onClick={() => setActiveTab('chat')}
                >
                  <Send size={14} />
                  <span>💬 Chatbot</span>
                </button>
                <button 
                  className={`dashboard-tab ${activeTab === 'rag' ? 'active' : ''}`}
                  onClick={() => setActiveTab('rag')}
                >
                  <Cpu size={14} />
                  <span>🧠 RAG Pipeline</span>
                </button>
              </div>

              {/* Left Pane: RAG Pipeline Telemetry Panel */}
              <div className={`rag-panel ${activeTab === 'rag' ? 'mobile-visible' : 'mobile-hidden'}`}>
                <div className="rag-panel-header">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Activity size={16} className="text-cyan-400 animate-pulse" />
                    <div>
                      <span className="rag-panel-title">Q-RAG-v1.2 Telemetry</span>
                      {activeQuery && (
                        <div style={{ fontSize: '0.65rem', color: 'var(--color-cyan)', marginTop: '0.1rem', fontFamily: 'var(--font-mono)' }}>
                          Query: "{activeQuery}"
                        </div>
                      )}
                    </div>
                  </div>
                  <span className="rag-telemetry-status font-mono">
                    {ragStep > 0 ? (
                      <span className="text-cyan-400 animate-pulse">● RUNNING</span>
                    ) : (
                      <span style={{ color: 'var(--text-muted)' }}>● STANDBY</span>
                    )}
                  </span>
                </div>

                <div className="rag-pipeline-flow">
                  {/* Step 1: Tokenizer */}
                  <div className={`rag-pipeline-step ${ragStep === 1 ? 'active' : ragStep > 1 ? 'completed' : ''}`}>
                    <div className="step-indicator">
                      <span className="step-num">1</span>
                    </div>
                    <div className="step-details">
                      <div className="step-name">Query Tokenizer</div>
                      <div className="step-content">
                        {ragStep === 0 && <span style={{ color: 'var(--text-muted)' }}>Awaiting question...</span>}
                        {ragStep >= 1 && (
                          <div className="tokens-wrap">
                            {processedTokens.map((t, idx) => (
                              <span key={idx} className="token-chip">{t}</span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Step 2: Dense Vector Embedding */}
                  <div className={`rag-pipeline-step ${ragStep === 2 ? 'active' : ragStep > 2 ? 'completed' : ''}`}>
                    <div className="step-indicator">
                      <span className="step-num">2</span>
                    </div>
                    <div className="step-details">
                      <div className="step-name">Vector Embedding (dense)</div>
                      <div className="step-content">
                        {ragStep < 2 && <span style={{ color: 'var(--text-muted)' }}>Awaiting vectorizer...</span>}
                        {ragStep >= 2 && (
                          <div className="vector-matrix-display font-mono">
                            <span className="vector-bracket">[</span>
                            {embeddingVector.map((v, idx) => (
                              <span key={idx} className="vector-val">
                                {v >= 0 ? `+${v.toFixed(3)}` : v.toFixed(3)}
                                {idx < embeddingVector.length - 1 ? ', ' : ''}
                              </span>
                            ))}
                            <span className="vector-bracket">]</span>
                            <div className="matrix-scanner"></div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Step 3: Vector & Keyword Search */}
                  <div className={`rag-pipeline-step ${ragStep === 3 ? 'active' : ragStep > 3 ? 'completed' : ''}`}>
                    <div className="step-indicator">
                      <span className="step-num">3</span>
                    </div>
                    <div className="step-details">
                      <div className="step-name">ChromaDB & BM25 Match</div>
                      <div className="step-content">
                        {ragStep < 3 && <span style={{ color: 'var(--text-muted)' }}>Awaiting DB scan...</span>}
                        {ragStep >= 3 && (
                          <div className="retrieved-docs-list">
                            {retrievedDocs.slice(0, 2).map((d, idx) => (
                              <div key={idx} className="retrieved-doc-item">
                                <div className="retrieved-doc-meta">
                                  <span className="retrieved-doc-title">{d.title}</span>
                                  <span className="retrieved-doc-score">{(d.score * 100).toFixed(1)}%</span>
                                </div>
                                <div className="retrieved-doc-body">{d.content}</div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Step 4: Reranker */}
                  <div className={`rag-pipeline-step ${ragStep === 4 ? 'active' : ragStep > 4 ? 'completed' : ''}`}>
                    <div className="step-indicator">
                      <span className="step-num">4</span>
                    </div>
                    <div className="step-details">
                      <div className="step-name">Cross-Encoder Reranker</div>
                      <div className="step-content">
                        {ragStep < 4 && <span style={{ color: 'var(--text-muted)' }}>Awaiting scoring...</span>}
                        {ragStep >= 4 && (
                          <div className="reranking-list">
                            {retrievedDocs.slice(0, 3).map((d, idx) => (
                              <div key={idx} className="rerank-item">
                                <div className="rerank-bar-container">
                                  <div className="rerank-bar" style={{ width: `${d.score * 100}%` }}></div>
                                </div>
                                <span className="rerank-title">{d.title}</span>
                                <span className="rerank-score font-mono">{d.score}</span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Step 5: Prompt Synthesis */}
                  <div className={`rag-pipeline-step ${ragStep === 5 ? 'active' : ''}`}>
                    <div className="step-indicator">
                      <span className="step-num">5</span>
                    </div>
                    <div className="step-details">
                      <div className="step-name">Synthesis Context Prompt</div>
                      <div className="step-content">
                        {ragStep < 5 && <span style={{ color: 'var(--text-muted)' }}>Awaiting generation template...</span>}
                        {ragStep >= 5 && (
                          <div className="synthesis-prompt-box font-mono">
                            <pre>{synthesisPrompt}</pre>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Right Pane: Q-AI Interactive Chatbot Panel */}
              <div className={`chatbot-panel ${activeTab === 'chat' ? 'mobile-visible' : 'mobile-hidden'}`}>
                <div className="chatbot-header">
                  <div className="chatbot-user-info">
                    <div className="chatbot-avatar">Q</div>
                    <div className="chatbot-meta">
                      <span className="chatbot-name">Q-AI Assistant</span>
                      <span className="chatbot-status">
                        <span className="chatbot-status-dot"></span>
                        Online
                      </span>
                    </div>
                  </div>
                  <Terminal size={18} style={{ color: 'var(--text-muted)' }} />
                </div>

                {/* AI Systems Metrics Monitor Bar */}
                <div className="ai-metrics-bar">
                  <div className="metric-item">
                    <span className="metric-dot purple"></span>
                    <span>Engine: <strong style={{ color: 'var(--color-cyan)' }}>Q-RAG-v1.2</strong></span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-dot green"></span>
                    <span>Inference: <strong>Ollama Local</strong></span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-dot blue"></span>
                    <span>Speed: <strong>14.2 t/s</strong></span>
                  </div>
                </div>

                <div className="chatbot-body">
                  <div className="chatbot-scanline"></div>
                  <div ref={messagesContainerRef} className="chatbot-messages">
                    {messages.map(msg => (
                      <div 
                        key={msg.id} 
                        className={`chat-message ${msg.sender === 'ai' ? 'message-ai' : 'message-user'}`}
                      >
                        {msg.text}
                      </div>
                    ))}
                    {isTyping && (
                      <div className="chat-message message-ai" style={{ opacity: 0.9, fontFamily: 'var(--font-mono)', fontSize: '0.78rem', color: 'var(--color-cyan)', display: 'flex', alignItems: 'center', gap: '0.5rem', borderLeft: '2px solid var(--color-cyan)' }}>
                        <span className="chatbot-status-dot"></span>
                        {ragStep === 1 && <span>🧠 RAG: Tokenizing query parameters...</span>}
                        {ragStep === 2 && <span>🧠 RAG: Projecting query into embedding space...</span>}
                        {ragStep === 3 && <span>🧠 RAG: Fetching sparse & dense matrix candidates...</span>}
                        {ragStep === 4 && <span>🧠 RAG: Reranking candidates with cross-entropy loss...</span>}
                        {ragStep === 5 && <span>🧠 RAG: Synthesizing customized prompt template...</span>}
                        {ragStep === 0 && <span>🧠 RAG: LLM generating citation-cited response...</span>}
                      </div>
                    )}
                  </div>

                  <div className="chat-suggestions-title">💡 Click to Ask Q-AI</div>
                  <div className="chat-suggestions-container">
                    <button 
                      onClick={() => handleSuggestionClick('Tell me about the SPC Roster Web App')} 
                      className="suggestion-chip"
                    >
                      🚀 SPC Roster Web App
                    </button>
                    <button 
                      onClick={() => handleSuggestionClick('Explain the WorkFlow AI RAG project')} 
                      className="suggestion-chip"
                    >
                      🧠 WorkFlow AI (RAG)
                    </button>
                    <button 
                      onClick={() => handleSuggestionClick('What is your visa & sponsorship status in Australia?')} 
                      className="suggestion-chip"
                    >
                      🇦🇺 Visa & Sponsorship
                    </button>
                    <button 
                      onClick={() => handleSuggestionClick('Show me Qs contact details')} 
                      className="suggestion-chip"
                    >
                      📞 How to Contact
                    </button>
                  </div>

                  <div className="chatbot-footer">
                    <form onSubmit={handleSendMessage} className="chatbot-input-container">
                      <input 
                        type="text" 
                        value={chatInput}
                        onChange={(e) => setChatInput(e.target.value)}
                        placeholder="Ask Q-AI a question about Q's resume..."
                        className="chatbot-input"
                      />
                      <button type="submit" className="chatbot-send-btn">
                        <Send size={16} />
                      </button>
                    </form>
                  </div>
                </div>
              </div>
            </div>

          </div>
        </div>
      </section>

      {/* Triple Threat Value Proposition Section */}
      <section className="glass" style={{ borderLeft: 'none', borderRight: 'none', background: 'rgba(255, 255, 255, 0.01)' }}>
        <div className="max-width-container">
          <div className="section-header">
            <span className="section-pretitle">The Unique Value Profile</span>
            <h2 className="section-title">Three Pillars of Expertise</h2>
          </div>

          <div className="threat-grid">
            {/* Value 1 */}
            <div className="threat-card glass glass-interactive">
              <div className="threat-icon-box cyan">
                <Code size={24} />
              </div>
              <h3 className="threat-title">Full Stack Web Engineering</h3>
              <p className="threat-desc">
                Proficient in React, React Native, TypeScript, Node.js, and FastAPI. Q builds highly responsive Single Page Applications connected to robust database backends (Supabase, Turso/LibSQL, MySQL). He ensures rapid, agile delivery in production-scale deployments.
              </p>
            </div>

            {/* Value 2 */}
            <div className="threat-card glass glass-interactive">
              <div className="threat-icon-box indigo">
                <Cpu size={24} />
              </div>
              <h3 className="threat-title">Advanced AI & RAG Orchestration</h3>
              <p className="threat-desc">
                Possesses a Bachelor\'s degree in Software Engineering (AI). Q builds sophisticated RAG systems incorporating local LLMs (Ollama), vector stores (Chroma DB), hybrid retrieval (Dense + Sparse BM25 RRF), reranking layers, and hallucinations controls.
              </p>
            </div>

            {/* Value 3 */}
            <div className="threat-card glass glass-interactive">
              <div className="threat-icon-box emerald">
                <Layers size={24} />
              </div>
              <h3 className="threat-title">Operations & Logistics Domain</h3>
              <p className="threat-desc">
                Combines technical software testing (Rubicon) with extensive logistics leadership (Premier Fresh Team Leader, SPC Forklift Driver). Q understands industrial warehouse workflows, SAP ERP systems, and manual bottlenecks, driving high-impact digital automation.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Interactive Projects Section */}
      <section id="projects">
        <div className="max-width-container">
          <div className="section-header">
            <span className="section-pretitle">Interactive Portfolio Showcase</span>
            <h2 className="section-title">Core Projects Archive</h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', maxWidth: '600px', marginTop: '0.5rem' }}>
              Select any tech tag in the **Skills Matrix** below to automatically highlight the corresponding project cards using that technology!
            </p>
          </div>

          <div className="project-grid">
            {projects.map(proj => {
              // Highlight code check
              const isProjHighlighted = selectedSkill ? proj.tech.includes(selectedSkill) : false;
              
              return (
                <div 
                  key={proj.id} 
                  className={`project-card glass glass-interactive ${isProjHighlighted ? 'highlighted' : ''}`}
                >
                  <div className="project-body">
                    <div className="project-header-row">
                      <div>
                        <h3 className="project-title">{proj.title}</h3>
                        <div style={{ color: 'var(--color-cyan)', fontSize: '0.85rem', fontWeight: 600, marginTop: '0.2rem' }}>
                          {proj.role}
                        </div>
                      </div>
                      <span className="project-period">{proj.period}</span>
                    </div>

                    <p className="project-description">{proj.description}</p>

                    <ul className="project-bullets">
                      {proj.bullets.map((bullet, idx) => (
                        <li key={idx}>{bullet}</li>
                      ))}
                    </ul>

                    <div className="project-tags">
                      {proj.tech.map((t, idx) => (
                        <span 
                          key={idx} 
                          className={`project-tag ${t === selectedSkill ? 'active' : ''}`}
                        >
                          {t}
                        </span>
                      ))}
                    </div>
                  </div>

                  {proj.impact && (
                    <div className="project-impact-banner">
                      <Sparkles size={14} />
                      <span>{proj.impact}</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Interactive Skills Matrix Section */}
      <section id="skills" className="glass" style={{ borderLeft: 'none', borderRight: 'none', background: 'rgba(3, 7, 18, 0.4)' }}>
        <div className="max-width-container">
          <div className="section-header">
            <span className="section-pretitle">Dynamic Skills Matrix</span>
            <h2 className="section-title">Skills Grid & Project Linker</h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', maxWidth: '600px', marginTop: '0.5rem' }}>
              Click any skill badge to discover which projects in the archive were designed using that tool!
            </p>
          </div>

          <div className="skills-container">
            {skillGroups.map((group, idx) => (
              <div key={idx} className="skills-category-card glass">
                <h3 className="skills-category-title">
                  {group.icon}
                  <span>{group.category}</span>
                </h3>
                <div className="skills-tags-wrap">
                  {group.skills.map((skill, sIdx) => {
                    const isActive = selectedSkill === skill;
                    return (
                      <button
                        key={sIdx}
                        onClick={() => handleSkillClick(skill)}
                        className={`matrix-skill-tag ${isActive ? 'active' : ''}`}
                      >
                        {skill}
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Interactive 2D Embedding Space Visualizer Section */}
      <section id="embedding" style={{ background: 'rgba(3, 7, 18, 0.5)' }}>
        <div className="max-width-container">
          <div className="section-header">
            <span className="section-pretitle">Neural Latent Space</span>
            <h2 className="section-title">Project & Skill Embedding Space</h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem', maxWidth: '650px', marginTop: '0.5rem' }}>
              Interact with the neural vector map! Click any node or search to trigger real-time **Cosine Similarity** calculations. Watch similar concepts gravitate to the center connected by glowing neon semantic lasers.
            </p>
          </div>

          <div className="embedding-container glass">
            {/* Visualizer Canvas Block */}
            <div className="canvas-wrapper">
              <EmbeddingSpaceVisualizer 
                activeFocusId={activeEmbeddingFocus} 
                onNodeClick={(id) => {
                  setActiveEmbeddingFocus(id);
                  addAgentLog('SYSTEM', `Focal node shifted to embedding cell: '${id}'`);
                }}
              />
            </div>

            {/* Sidebar Details Panel */}
            <div className="vector-details-panel">
              <div className="panel-hud-header">
                <Terminal size={14} style={{ color: 'var(--color-cyan)' }} />
                <span className="panel-hud-title">Vector Engine Readout</span>
              </div>

              {activeEmbeddingFocus ? (
                <div className="hud-readout-active">
                  <div className="readout-group">
                    <span className="readout-label">ACTIVE NODE</span>
                    <span className="readout-value active-node-label">
                      {activeEmbeddingFocus.toUpperCase()}
                    </span>
                  </div>

                  <div className="readout-group">
                    <span className="readout-label">LATENT SPACE COORDINATES</span>
                    <span className="readout-value font-mono text-cyan">
                      [{vectorData[activeEmbeddingFocus] ? vectorData[activeEmbeddingFocus].map(v => v.toFixed(3)).join(', ') : '0, 0, 0, 0'}]
                    </span>
                  </div>

                  <div className="readout-group">
                    <span className="readout-label">NEURAL MATRIX CONNECTIONS</span>
                    <div className="connections-list font-mono">
                      {Object.keys(vectorData)
                        .filter(id => id !== activeEmbeddingFocus)
                        .map(id => {
                          const similarity = getCosineSimilarity(vectorData[activeEmbeddingFocus], vectorData[id]);
                          return { id, similarity };
                        })
                        .filter(item => item.similarity > 0.65)
                        .sort((a, b) => b.similarity - a.similarity)
                        .slice(0, 5)
                        .map(item => (
                          <div key={item.id} className="connection-item">
                            <span className="connect-name">→ {item.id}</span>
                            <span className="connect-score">
                              {(item.similarity * 100).toFixed(1)}% Similarity
                            </span>
                          </div>
                        ))}
                    </div>
                  </div>

                  <button 
                    onClick={() => {
                      setActiveEmbeddingFocus(null);
                      addAgentLog('SYSTEM', 'Cleared Latent Space vectors filter.');
                    }}
                    className="btn-secondary clear-focus-btn"
                  >
                    Reset Latent Space
                  </button>
                </div>
              ) : (
                <div className="hud-readout-idle font-mono text-muted">
                  <span className="blink-text">_ Awaiting vector anchor...</span>
                  <p style={{ fontSize: '0.78rem', marginTop: '1rem', lineHeight: '1.4' }}>
                    Click any node floating inside the nebula or click a badge in the Skills Grid above to project its sparse multidimensional coordinate array.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Career & Growth Timeline Section */}
      <section id="timeline">
        <div className="max-width-container">
          <div className="section-header">
            <span className="section-pretitle">Career & Growth Journey</span>
            <h2 className="section-title">Growth & Experience Timeline</h2>
          </div>

          <div className="timeline-wrapper">
            {timeline.map((item, idx) => (
              <div key={idx} className="timeline-card glass glass-interactive">
                <div className="timeline-dot"></div>
                
                <div className="timeline-header">
                  <div className="timeline-title-row">
                    <span className="timeline-role">{item.role}</span>
                    <span className="timeline-company">
                      {item.company} <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>• {item.location}</span>
                    </span>
                  </div>
                  <span className="timeline-date">{item.period}</span>
                </div>

                <ul className="timeline-bullets">
                  {item.bullets.map((b, bIdx) => (
                    <li key={bIdx}>{b}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Contact & Recruitment Opportunity Section */}
      <section id="contact" style={{ background: 'rgba(3, 7, 18, 0.7)' }}>
        <div className="max-width-container">
          <div className="section-header">
            <span className="section-pretitle">Get In Touch</span>
            <h2 className="section-title">Opportunities & Coffee Chats</h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', maxWidth: '600px', marginTop: '0.5rem' }}>
              Q is open to full-time opportunities, interview requests, technical discussions, and corporate sponsorship chats.
            </p>
          </div>

          <div className="contact-grid">
            
            {/* Info Cards */}
            <div className="contact-info-card glass">
              <h3 style={{ fontSize: '1.25rem' }}>Personal Contact Information</h3>
              
              <div className="contact-meta-list">
                {/* Email (with copying interaction) */}
                <div className="contact-meta-item">
                  <div className="contact-icon-wrapper">
                    <Mail size={18} />
                  </div>
                  <div className="contact-meta-content">
                    <span className="contact-meta-label">Email Address</span>
                    <button 
                      onClick={handleCopyEmail}
                      className="contact-meta-value"
                      style={{ textAlign: 'left', textDecoration: 'underline' }}
                    >
                      {copiedEmail ? '✅ Copied to Clipboard!' : 'hyungkyu.lee.q@gmail.com'}
                    </button>
                  </div>
                </div>

                {/* Location */}
                <div className="contact-meta-item">
                  <div className="contact-icon-wrapper">
                    <MapPin size={18} />
                  </div>
                  <div className="contact-meta-content">
                    <span className="contact-meta-label">Primary Location</span>
                    <span className="contact-meta-value">Shepparton, Victoria, Australia (Relocation OK)</span>
                  </div>
                </div>

                {/* LinkedIn */}
                <div className="contact-meta-item">
                  <div className="contact-icon-wrapper">
                    <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path><rect width="4" height="12" x="2" y="9"></rect><circle cx="4" cy="4" r="2"></circle></svg>
                  </div>
                  <div className="contact-meta-content">
                    <span className="contact-meta-label">LinkedIn</span>
                    <a 
                      href="https://linkedin.com/in/qleeq" 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="contact-meta-value"
                      style={{ display: 'flex', alignItems: 'center', gap: '0.2rem' }}
                    >
                      <span>linkedin.com/in/qleeq</span>
                      <ExternalLink size={12} />
                    </a>
                  </div>
                </div>

                {/* GitHub */}
                <div className="contact-meta-item">
                  <div className="contact-icon-wrapper">
                    <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"></path><path d="M9 18c-4.51 2-5-2-7-2"></path></svg>
                  </div>
                  <div className="contact-meta-content">
                    <span className="contact-meta-label">GitHub Repository</span>
                    <a 
                      href="https://github.com/Q-Leee" 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="contact-meta-value"
                      style={{ display: 'flex', alignItems: 'center', gap: '0.2rem' }}
                    >
                      <span>github.com/Q-Leee</span>
                      <ExternalLink size={12} />
                    </a>
                  </div>
                </div>
              </div>
            </div>

            {/* Static Interactive Form */}
            <div className="contact-form-card glass">
              <h3 style={{ fontSize: '1.25rem' }}>Send a Message Instantly</h3>
              <p style={{ color: 'var(--color-cyan)', fontWeight: '700', fontSize: '0.92rem', marginTop: '0.25rem', marginBottom: '0.75rem', textShadow: '0 0 10px rgba(57, 255, 20, 0.4)' }}>
                Send me a job offer. I will personally review it and get back to you! 😉
              </p>
              
              <form onSubmit={async (e) => {
                e.preventDefault();
                
                const endpoint = 'https://api.web3forms.com/submit';
                const accessKey = '219152e5-a352-45dc-868d-b27aa33a4891';

                setContactSending(true);
                addAgentLog('SYSTEM', 'Initiating secure email dispatch protocol...');
                addAgentLog('SYSTEM', 'Target endpoint resolved: Web3Forms Secure Gateway');

                try {
                  const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: {
                      'Content-Type': 'application/json',
                      'Accept': 'application/json'
                    },
                    body: JSON.stringify({
                      access_key: accessKey,
                      name: contactName,
                      email: contactEmail,
                      subject: contactSubject,
                      message: contactMessage
                    })
                  });

                  const data = await response.json();

                  if (response.ok && data.success) {
                    setModalContent({
                      title: 'TRANSMISSION SECURE',
                      message: 'Your message has been successfully dispatched to Q\'s personal email!',
                      subtext: 'I have logged this recruitment interaction. Q will personally review the details and respond to your return address shortly. Thank you!'
                    });
                    addAgentLog('SYSTEM', `Successfully emailed message from '${contactName}' to Q.`);
                    setContactName('');
                    setContactEmail('');
                    setContactSubject('');
                    setContactMessage('');
                  } else {
                    throw new Error(data.message || `Web3Forms response not OK (Status: ${response.status})`);
                  }
                } catch (error) {
                  const errorMsg = error instanceof Error ? error.message : 'Unknown Error';
                  setModalContent({
                    title: 'TRANSMISSION TIMEOUT',
                    message: 'The neural transmission gateway encountered a connection latency error.',
                    subtext: 'Could not deliver the message automatically. Please feel free to copy your text and email Q directly at: hyungkyu.lee.q@gmail.com!'
                  });
                  addAgentLog('SYSTEM', `Email gateway transmission failed: ${errorMsg}. Fallback active.`);
                } finally {
                  setContactSending(false);
                  setShowModal(true);
                }
              }} className="contact-form-card" style={{ padding: 0, background: 'none', border: 'none', boxShadow: 'none' }}>
                
                <div className="form-group-row">
                  <div className="form-group">
                    <label className="form-label">Your Name & Company</label>
                    <input 
                      type="text" 
                      required 
                      placeholder="e.g. Hiring Manager at Google" 
                      className="form-input" 
                      value={contactName}
                      onChange={(e) => setContactName(e.target.value)}
                      disabled={contactSending}
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Return Email Address</label>
                    <input 
                      type="email" 
                      required 
                      placeholder="contact@company.com" 
                      className="form-input" 
                      value={contactEmail}
                      onChange={(e) => setContactEmail(e.target.value)}
                      disabled={contactSending}
                    />
                  </div>
                </div>

                <div className="form-group">
                  <label className="form-label">Subject</label>
                  <input 
                    type="text" 
                    required 
                    placeholder="e.g. Software Engineer Opportunity" 
                    className="form-input" 
                    value={contactSubject}
                    onChange={(e) => setContactSubject(e.target.value)}
                    disabled={contactSending}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Message Details</label>
                  <textarea 
                    rows={4} 
                    required 
                    placeholder="Write details about the opening, relocation, or coffee chat proposal here..." 
                    className="form-input" 
                    style={{ resize: 'vertical' }}
                    value={contactMessage}
                    onChange={(e) => setContactMessage(e.target.value)}
                    disabled={contactSending}
                  ></textarea>
                </div>

                <button 
                  type="submit" 
                  className="btn-primary" 
                  style={{ alignSelf: 'flex-start', marginTop: '0.5rem', opacity: contactSending ? 0.7 : 1 }}
                  disabled={contactSending}
                >
                  <Send size={16} className={contactSending ? "animate-pulse" : ""} />
                  <span>{contactSending ? "Sending Securely..." : "Send Message"}</span>
                </button>

              </form>
            </div>

          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="max-width-container">
          <p>© {new Date().getFullYear()} HyungKyu (Q) Lee. All Rights Reserved.</p>
          <p style={{ marginTop: '0.5rem', fontSize: '0.75rem', opacity: 0.6 }}>
            Designed & Engineered with React + Vite + TypeScript. Deployed on Vercel.
          </p>
        </div>
      </footer>
      {/* Holographic Alert Modal */}
      {showModal && (
        <div className="custom-modal-overlay" onClick={() => setShowModal(false)}>
          <div className="custom-modal-card glass" onClick={(e) => e.stopPropagation()}>
            <div className="custom-modal-scanline"></div>
            <div className="custom-modal-glow"></div>
            
            <div className="custom-modal-header">
              <Sparkles size={22} style={{ color: 'var(--color-cyan)' }} className="animate-pulse" />
              <h3 className="custom-modal-title">{modalContent.title}</h3>
            </div>
            
            <div className="custom-modal-body">
              <p className="custom-modal-message">{modalContent.message}</p>
              {modalContent.subtext && (
                <p className="custom-modal-subtext">{modalContent.subtext}</p>
              )}
            </div>
            
            <div className="custom-modal-footer">
              <button 
                onClick={() => setShowModal(false)}
                className="btn-primary custom-modal-btn"
              >
                <span>Acknowledge</span>
              </button>
            </div>
          </div>
        </div>
      )}
      {/* Collapsible Cyber Agent Telemetry HUD */}
      <div className={`agent-hud-container ${hudCollapsed ? 'collapsed' : ''}`}>
        {hudCollapsed ? (
          <button 
            onClick={() => setHudCollapsed(false)}
            className="agent-hud-collapsed-trigger glass glass-interactive"
          >
            <Terminal size={14} style={{ color: 'var(--color-cyan)' }} className="animate-pulse" />
            <span>● AGENT_HUD: ACTIVE</span>
          </button>
        ) : (
          <div className="agent-hud-panel glass">
            <div className="agent-hud-header">
              <div className="agent-hud-status">
                <span className="agent-hud-dot"></span>
                <span className="agent-hud-title">Q-AGENT TELEMETRY HUD</span>
              </div>
              <button 
                onClick={() => setHudCollapsed(true)}
                className="agent-hud-minimize"
              >
                —
              </button>
            </div>
            
            <div ref={logsContainerRef} className="agent-hud-console">
              {agentLogs.map(log => (
                <div key={log.id} className="hud-log-line">
                  <span className="hud-log-time">{log.timestamp}</span>
                  <span className={`hud-log-cat cat-${log.category.toLowerCase()}`}>
                    [{log.category}]
                  </span>
                  <span className="hud-log-text">{log.text}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
