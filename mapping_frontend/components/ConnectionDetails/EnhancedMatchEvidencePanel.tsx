// components/ConnectionDetails/MatchEvidencePanel.tsx
import React from 'react';
import { 
  MatchValueData, 
  VisualizationEntityEdge,
  EmailMatchValue,
  NameMatchValue,
  UrlMatchValue,
  PhoneMatchValue,
  AddressMatchValue
} from '../../lib/types';
import { 
  getConfidenceTextColor, 
  getConfidenceBgColor, 
  formatConfidence, 
  capitalize
} from '../../lib/visualization-utils';

interface MatchEvidencePanelProps {
  connection: VisualizationEntityEdge;
}

const MatchEvidencePanel: React.FC<MatchEvidencePanelProps> = ({ connection }) => {
  const renderMethodEvidence = (method: {
    method_type: string;
    pre_rl_confidence: number;
    rl_confidence: number;
    combined_confidence: number;
    match_values?: MatchValueData;
  }, index: number) => {
    // Common header for all method types
    const methodHeader = (
      <div className="flex justify-between items-center mb-2">
        <h4 className="font-medium text-gray-800">
          {capitalize(method.method_type)} Match
        </h4>
        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getConfidenceBgColor(method.combined_confidence)} ${getConfidenceTextColor(method.combined_confidence)}`}>
          {formatConfidence(method.combined_confidence)} Confidence
        </span>
      </div>
    );
    
    // Common confidence metrics display
    const confidenceMetrics = (
      <div className="grid grid-cols-2 gap-2 text-sm mb-2">
        <div>
          <span className="text-gray-600">Pre-ML Confidence:</span>
          <span className={`ml-1 font-medium ${getConfidenceTextColor(method.pre_rl_confidence)}`}>
            {formatConfidence(method.pre_rl_confidence)}
          </span>
        </div>
        
        <div>
          <span className="text-gray-600">ML Enhanced:</span>
          <span className={`ml-1 font-medium ${getConfidenceTextColor(method.rl_confidence)}`}>
            {formatConfidence(method.rl_confidence)}
          </span>
        </div>
      </div>
    );
    
    // Method-specific evidence rendering
    let methodEvidence = null;
    
    // Check if match_values exists and handle based on method type
    if (method.match_values) {
      switch (method.method_type) {
        case 'email':
          if (method.match_values.type === 'email') {
            const emailValues = method.match_values.values as EmailMatchValue;
            methodEvidence = (
              <div className="bg-gray-50 p-2 rounded text-sm">
                <div className="grid grid-cols-2 gap-1">
                  <div>
                    <div className="text-gray-600 text-xs">Original Email 1:</div>
                    <div className="font-mono">{emailValues.original_email1 || 'N/A'}</div>
                  </div>
                  <div>
                    <div className="text-gray-600 text-xs">Original Email 2:</div>
                    <div className="font-mono">{emailValues.original_email2 || 'N/A'}</div>
                  </div>
                </div>
                {emailValues.normalized_shared_email && (
                  <div className="mt-2">
                    <div className="text-gray-600 text-xs">Normalized Email:</div>
                    <div className="font-mono font-medium">{emailValues.normalized_shared_email}</div>
                  </div>
                )}
              </div>
            );
          }
          break;
          
        case 'name':
          if (method.match_values.type === 'name') {
            const nameValues = method.match_values.values as NameMatchValue;
            methodEvidence = (
              <div className="bg-gray-50 p-2 rounded text-sm">
                <div className="grid grid-cols-2 gap-1">
                  <div>
                    <div className="text-gray-600 text-xs">Original Name 1:</div>
                    <div>{nameValues.original_name1 || 'N/A'}</div>
                  </div>
                  <div>
                    <div className="text-gray-600 text-xs">Original Name 2:</div>
                    <div>{nameValues.original_name2 || 'N/A'}</div>
                  </div>
                </div>
                {nameValues.pre_rl_match_type && (
                  <div className="mt-2">
                    <div className="text-gray-600 text-xs">Match Type:</div>
                    <div className="font-medium">{nameValues.pre_rl_match_type}</div>
                  </div>
                )}
              </div>
            );
          }
          break;
          
        case 'url':
          if (method.match_values.type === 'url') {
            const urlValues = method.match_values.values as UrlMatchValue;
            methodEvidence = (
              <div className="bg-gray-50 p-2 rounded text-sm">
                <div className="grid grid-cols-2 gap-1">
                  <div>
                    <div className="text-gray-600 text-xs">Original URL 1:</div>
                    <div className="font-mono break-all text-xs">{urlValues.original_url1 || 'N/A'}</div>
                  </div>
                  <div>
                    <div className="text-gray-600 text-xs">Original URL 2:</div>
                    <div className="font-mono break-all text-xs">{urlValues.original_url2 || 'N/A'}</div>
                  </div>
                </div>
                {urlValues.normalized_shared_domain && (
                  <div className="mt-2">
                    <div className="text-gray-600 text-xs">Normalized Domain:</div>
                    <div className="font-mono font-medium text-xs">{urlValues.normalized_shared_domain}</div>
                  </div>
                )}
                {urlValues.matching_slug_count > 0 && (
                  <div className="mt-2">
                    <div className="text-gray-600 text-xs">Matching Slug Count:</div>
                    <div className="font-medium">{urlValues.matching_slug_count}</div>
                  </div>
                )}
              </div>
            );
          }
          break;
          
        case 'phone':
          if (method.match_values.type === 'phone') {
            const phoneValues = method.match_values.values as PhoneMatchValue;
            methodEvidence = (
              <div className="bg-gray-50 p-2 rounded text-sm">
                <div className="grid grid-cols-2 gap-1">
                  <div>
                    <div className="text-gray-600 text-xs">Original Phone 1:</div>
                    <div className="font-mono">{phoneValues.original_phone1 || 'N/A'}</div>
                  </div>
                  <div>
                    <div className="text-gray-600 text-xs">Original Phone 2:</div>
                    <div className="font-mono">{phoneValues.original_phone2 || 'N/A'}</div>
                  </div>
                </div>
                {phoneValues.normalized_shared_phone && (
                  <div className="mt-2">
                    <div className="text-gray-600 text-xs">Normalized Phone:</div>
                    <div className="font-mono font-medium">{phoneValues.normalized_shared_phone}</div>
                  </div>
                )}
                {(phoneValues.extension1 || phoneValues.extension2) && (
                  <div className="mt-2 grid grid-cols-2 gap-1">
                    <div>
                      <div className="text-gray-600 text-xs">Extension 1:</div>
                      <div className="font-mono">{phoneValues.extension1 || 'N/A'}</div>
                    </div>
                    <div>
                      <div className="text-gray-600 text-xs">Extension 2:</div>
                      <div className="font-mono">{phoneValues.extension2 || 'N/A'}</div>
                    </div>
                  </div>
                )}
              </div>
            );
          }
          break;
          
        case 'address':
          if (method.match_values.type === 'address') {
            const addressValues = method.match_values.values as AddressMatchValue;
            methodEvidence = (
              <div className="bg-gray-50 p-2 rounded text-sm">
                <div className="grid grid-cols-2 gap-1">
                  <div>
                    <div className="text-gray-600 text-xs">Original Address 1:</div>
                    <div>{addressValues.original_address1 || 'N/A'}</div>
                  </div>
                  <div>
                    <div className="text-gray-600 text-xs">Original Address 2:</div>
                    <div>{addressValues.original_address2 || 'N/A'}</div>
                  </div>
                </div>
                {addressValues.normalized_shared_address && (
                  <div className="mt-2">
                    <div className="text-gray-600 text-xs">Normalized Address:</div>
                    <div className="font-medium">{addressValues.normalized_shared_address}</div>
                  </div>
                )}
                {addressValues.pairwise_match_score !== undefined && (
                  <div className="mt-2">
                    <div className="text-gray-600 text-xs">Match Score:</div>
                    <div className="font-medium">
                      {formatConfidence(addressValues.pairwise_match_score)}
                    </div>
                  </div>
                )}
              </div>
            );
          }
          break;
          
        default:
          // Default case for any other method type - show all values
          methodEvidence = (
            <div className="bg-gray-50 p-2 rounded">
              <div className="text-xs space-y-1">
                {Object.entries(method.match_values.values).map(([key, value]) => (
                  <div key={key} className="grid grid-cols-2">
                    <span className="text-gray-600">{key}:</span>
                    <span className="font-mono break-all">
                      {typeof value === 'object' 
                        ? JSON.stringify(value)
                        : String(value || 'N/A')}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          );
      }
    }
    
    return (
      <div key={`${method.method_type}-${index}`} className="border rounded-md p-3 mb-2 last:mb-0">
        {methodHeader}
        {confidenceMetrics}
        {methodEvidence || (
          <div className="text-sm text-gray-500 italic">
            No detailed evidence available for this match method.
          </div>
        )}
      </div>
    );
  };
  
  return (
    <div>
      <div className="flex justify-between items-center mb-4">
        <h3 className="font-semibold text-gray-900">Match Assessment</h3>
        <div className="flex items-center">
          <span className="text-gray-700 mr-2">Overall Confidence:</span>
          <span className={`px-2 py-1 rounded-full font-bold ${getConfidenceBgColor(connection.edge_weight)} ${getConfidenceTextColor(connection.edge_weight)}`}>
            {formatConfidence(connection.edge_weight)}
          </span>
        </div>
      </div>
      
      <div className="mb-2 text-sm text-gray-500">
        <p>This connection was determined using {connection.details.method_count} matching method{connection.details.method_count !== 1 ? 's' : ''}.</p>
      </div>
      
      <div className="mt-3">
        {connection.details.methods.map((method, index) => renderMethodEvidence(method, index))}
      </div>
    </div>
  );
};

export default MatchEvidencePanel;