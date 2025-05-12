// components/GraphVisualization/LinkTooltip.tsx
import React from 'react';
import { EntityGroup } from '../../lib/types';

interface LinkTooltipProps {
  link: EntityGroup;
  position: { x: number; y: number };
}

const LinkTooltip: React.FC<LinkTooltipProps> = ({ link, position }) => {
  if (!link) return null;
  
  // Helper function to safely render values (handle empty objects and nulls)
  const safeRender = (value: unknown): string => {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value === 'object' && Object.keys(value).length === 0) return 'N/A';
    return String(value);
  };
  
  // Render different evidence based on method type
  const renderEvidence = () => {
    const { match_values } = link;
    
    if (!match_values || !match_values.values) {
      return <div>No match evidence available</div>;
    }
    
    switch (link.method_type) {
      case 'email':
        return (
          <>
            <div className="mt-1">
              <span className="font-medium">Original Email 1:</span> {safeRender(match_values.values.original_email1)}
            </div>
            <div className="mt-1">
              <span className="font-medium">Original Email 2:</span> {safeRender(match_values.values.original_email2)}
            </div>
            <div className="mt-1">
              <span className="font-medium">Normalized Email:</span> {safeRender(match_values.values.normalized_shared_email)}
            </div>
          </>
        );
        
      case 'name':
        return (
          <>
            <div className="mt-1">
              <span className="font-medium">Original Name 1:</span> {safeRender(match_values.values.original_name1)}
            </div>
            <div className="mt-1">
              <span className="font-medium">Original Name 2:</span> {safeRender(match_values.values.original_name2)}
            </div>
            <div className="mt-1">
              <span className="font-medium">Similarity Score:</span> {safeRender(match_values.values.pre_rl_similarity_score)}
            </div>
          </>
        );
        
      case 'url':
        return (
          <>
            <div className="mt-1">
              <span className="font-medium">Original URL 1:</span> {safeRender(match_values.values.original_url1)}
            </div>
            <div className="mt-1">
              <span className="font-medium">Original URL 2:</span> {safeRender(match_values.values.original_url2)}
            </div>
            <div className="mt-1">
              <span className="font-medium">Normalized URL:</span> {safeRender(match_values.values.normalized_shared_url)}
            </div>
          </>
        );
        
      case 'phone':
        return (
          <>
            <div className="mt-1">
              <span className="font-medium">Original Phone 1:</span> {safeRender(match_values.values.original_phone1)}
            </div>
            <div className="mt-1">
              <span className="font-medium">Original Phone 2:</span> {safeRender(match_values.values.original_phone2)}
            </div>
            <div className="mt-1">
              <span className="font-medium">Normalized Phone:</span> {safeRender(match_values.values.normalized_shared_phone)}
            </div>
          </>
        );
        
      default:
        return (
          <div className="mt-1">
            <pre className="text-xs">{JSON.stringify(match_values.values, null, 2)}</pre>
          </div>
        );
    }
  };
  
  return (
    <div 
      className="absolute z-10 p-3 bg-white border border-gray-300 rounded-md shadow-lg"
      style={{
        left: position.x + 10, 
        top: position.y + 10,
        maxWidth: '300px'
      }}
    >
      <h3 className="font-bold text-md">Link Details</h3>
      <div className="mt-2 text-sm space-y-1">
        <div>
          <span className="font-medium">Method:</span> {link.method_type}
        </div>
        <div>
          <span className="font-medium">Confidence:</span> {link.confidence_score.toFixed(2)}
        </div>
        
        <div className="mt-2">
          <span className="font-medium">Evidence:</span>
          {renderEvidence()}
        </div>
      </div>
    </div>
  );
};

export default LinkTooltip;